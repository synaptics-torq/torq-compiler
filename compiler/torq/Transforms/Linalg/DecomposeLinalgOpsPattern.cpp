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
#include "torq/Utils/ShapeUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-decompose-linalg-ops-pattern"

using namespace std;

namespace mlir::syna::torq {

// %91 = linalg.generic {...} ins(%87 : tensor<1x2x128x1xbf16>) outs(%90 : tensor<1x2x128x1xi1>) {
// ^bb0(%in: bf16, %out: i1):
// %1484 = arith.extf %in : bf16 to f32
// %1485 = arith.cmpf olt, %1484, %cst_78 : f32
// linalg.yield %1485 : i1
// } -> tensor<1x2x128x1xi1>
// Detect a linalg.generic that:
//   1) extending each bf16 element to f32 (arith.extf)
//   2) comparing that f32 to a constant f32 (arith.cmpf)
//   3) yielding the i1 result.
// If the structure is exactly same, we rewrite it to remove the f32
// extension. We instead compare the original bf16 directly against a bf16
// constant (converted from the original f32 constant). This avoids f32 compare
// which the hardware does not support.
class CanonicalizeBF16CastComparePattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    CanonicalizeBF16CastComparePattern(MLIRContext *context)
        : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/0) {
        setDebugName("CanonicalizeBF16CastComparePattern");
    }

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {
        // Expect exactly 1 input tensor (bf16), 1 output tensor (i1).
        if (srcOp.getInputs().size() != 1 || srcOp.getOutputs().size() != 1)
            return rewriter.notifyMatchFailure(srcOp, "expected 1 input and 1 output");
        auto inType = dyn_cast<RankedTensorType>(srcOp.getInputs()[0].getType());
        auto outType = dyn_cast<RankedTensorType>(srcOp.getOutputs()[0].getType());
        if (!inType || !outType)
            return rewriter.notifyMatchFailure(srcOp, "expected ranked tensor types");
        if (!inType.getElementType().isBF16())
            return rewriter.notifyMatchFailure(srcOp, "expected bf16 input element type");
        if (!outType.getElementType().isInteger(1))
            return rewriter.notifyMatchFailure(srcOp, "expected i1 output element type");

        // Region must have: extf(bf16->f32), cmpf(f32,f32 with constant), yield(i1).
        auto &block = srcOp.getRegion().front();
        if (block.getNumArguments() < 1)
            return rewriter.notifyMatchFailure(srcOp, "expected block arg");
        // Must be exactly 3 ops.
        if (std::distance(block.begin(), block.end()) != 3)
            return rewriter.notifyMatchFailure(srcOp, "unexpected region ops count");

        auto *extOpRaw = &block.front();
        auto extOp = dyn_cast<arith::ExtFOp>(extOpRaw);
        if (!extOp)
            return rewriter.notifyMatchFailure(srcOp, "first op not arith.extf");
        if (!extOp.getIn().getType().isBF16() || !extOp.getResult().getType().isF32())
            return rewriter.notifyMatchFailure(srcOp, "extf types not bf16->f32");
        if (extOp.getIn() != block.getArgument(0))
            return rewriter.notifyMatchFailure(srcOp, "extf input not block argument");

        auto *cmpOpRaw = extOpRaw->getNextNode();
        auto cmpOp = dyn_cast<arith::CmpFOp>(cmpOpRaw);
        if (!cmpOp)
            return rewriter.notifyMatchFailure(srcOp, "second op not arith.cmpf");

        // One operand must be the extf result, the other a constant f32.
        arith::ConstantOp constOp = nullptr;
        Value bfExtendedVal;
        if (cmpOp.getLhs() == extOp.getResult()) {
            bfExtendedVal = cmpOp.getLhs();
            constOp = cmpOp.getRhs().getDefiningOp<arith::ConstantOp>();
        }
        else if (cmpOp.getRhs() == extOp.getResult()) {
            bfExtendedVal = cmpOp.getRhs();
            constOp = cmpOp.getLhs().getDefiningOp<arith::ConstantOp>();
        }
        else {
            return rewriter.notifyMatchFailure(srcOp, "cmpf does not use extf result");
        }
        if (!constOp)
            return rewriter.notifyMatchFailure(srcOp, "non-constant compare operand");
        auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue());
        if (!floatAttr || !floatAttr.getType().isF32())
            return rewriter.notifyMatchFailure(srcOp, "expected f32 constant rhs/lhs");

        auto *yieldOpRaw = cmpOpRaw->getNextNode();
        auto yieldOp = dyn_cast<linalg::YieldOp>(yieldOpRaw);
        if (!yieldOp)
            return rewriter.notifyMatchFailure(srcOp, "third op not linalg.yield");
        if (yieldOp.getValues().size() != 1 || yieldOp.getOperand(0) != cmpOp.getResult())
            return rewriter.notifyMatchFailure(srcOp, "yield not from cmpf result");

        // All checks passed. Replace with bf16 compare (no extf).
        auto bf16Ty = inType.getElementType(); // bf16
        double cstVal = floatAttr.getValueAsDouble();
        auto pred = cmpOp.getPredicate();

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            srcOp, srcOp.getResultTypes(), srcOp.getInputs(), srcOp.getOutputs(),
            srcOp.getIndexingMapsArray(), srcOp.getIteratorTypesArray(),
            [&](OpBuilder &b, Location loc, ValueRange args) {
                // Create bf16 constant.
                auto bfConst =
                    arith::ConstantOp::create(b, loc, b.getFloatAttr(bf16Ty, cstVal)).getResult();
                // Compare directly in bf16.
                auto newCmp = arith::CmpFOp::create(b, loc, pred, args[0], bfConst).getResult();
                linalg::YieldOp::create(b, loc, ValueRange{newCmp});
            }
        );

        return success();
    }
};

// Rewrites linalg.generic ops containing math.powf(x, 2) or math.fpowi(x, 2)
// (for bf16, i8, i16, i32 element types) into a linalg.generic with a multiplication
// (mul) of the input with itself. This is useful for lowering pow(x, 2) to mul(x, x)
// until the target dialect does not support powf/fpowi.
// Handles inline arith.constant exponents, constant tensor inputs,
// and math.fpowi with integer exponents (torch-mlir lowering).
class PowToMulPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    PowToMulPattern(MLIRContext *context)
        : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/0) {
        setDebugName("PowToMulPattern");
    }

  private:
    struct ExponentMatch {
        bool isInputTensor = false;
        unsigned inputIdx = 0;
    };

    // Check whether an attribute represents the value 2 for the given type.
    static bool attrIsTwo(Attribute attr, Type ty) {
        if (ty.isBF16()) {
            if (auto f = dyn_cast<FloatAttr>(attr))
                return f.getValueAsDouble() == 2.0;
        }
        else if (ty.isInteger(8) || ty.isInteger(16) || ty.isInteger(32)) {
            if (auto i = dyn_cast<IntegerAttr>(attr))
                return i.getInt() == 2;
        }
        return false;
    }

    // Try to match the exponent operand as the constant 2.
    // On success, populates `match` with tensor-input metadata.
    LogicalResult matchExponentTwo(
        linalg::GenericOp srcOp, math::PowFOp powfOp, math::FPowIOp fpowiOp, ExponentMatch &match,
        PatternRewriter &rewriter
    ) const {
        if (fpowiOp) {
            auto power = fpowiOp.getRhs();
            if (auto cOp = power.getDefiningOp<arith::ConstantOp>())
                if (auto intAttr = dyn_cast<IntegerAttr>(cOp.getValue()))
                    if (intAttr.getInt() == 2)
                        return success();
            return rewriter.notifyMatchFailure(srcOp, "fpowi exponent is not constant 2");
        }

        auto rhs = powfOp.getRhs();
        Attribute rhsAttr;

        // Case 1: RHS is an inline arith.constant in the body
        if (auto cOp = rhs.getDefiningOp<arith::ConstantOp>()) {
            rhsAttr = cOp.getValue();
        }
        // Case 2: RHS is a block argument from a constant tensor input
        else if (auto blockArg = dyn_cast<BlockArgument>(rhs)) {
            match.inputIdx = blockArg.getArgNumber();
            if (match.inputIdx < srcOp.getNumDpsInputs()) {
                Value inputTensor = srcOp.getInputs()[match.inputIdx];
                if (auto constOp = inputTensor.getDefiningOp<arith::ConstantOp>()) {
                    if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
                        if (denseAttr.isSplat()) {
                            rhsAttr = denseAttr.getSplatValue<Attribute>();
                            match.isInputTensor = true;
                        }
                    }
                }
            }
            if (!rhsAttr)
                return rewriter.notifyMatchFailure(
                    srcOp, "RHS block arg is not from a splat constant tensor"
                );
        }
        else {
            return rewriter.notifyMatchFailure(srcOp, "RHS is not a constant");
        }

        Type rhsType = rhs.getType();
        if (rhsType.isF32())
            return rewriter.notifyMatchFailure(srcOp, "Type f32 not supported");

        if (!attrIsTwo(rhsAttr, rhsType))
            return rewriter.notifyMatchFailure(srcOp, "RHS constant is not 2");

        return success();
    }

  public:
    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {
        // Match elementwise ops containing math.powf or math.fpowi
        Operation *binaryOp = getElementwiseBinaryOp(srcOp, /*allowConstant=*/true);
        if (!binaryOp)
            return rewriter.notifyMatchFailure(srcOp, "Not an elementwise binary op");

        auto powfOp = dyn_cast<math::PowFOp>(binaryOp);
        auto fpowiOp = dyn_cast<math::FPowIOp>(binaryOp);
        if (!powfOp && !fpowiOp)
            return rewriter.notifyMatchFailure(srcOp, "Not a math.powf or math.fpowi op");

        ExponentMatch expMatch;
        if (failed(matchExponentTwo(srcOp, powfOp, fpowiOp, expMatch, rewriter)))
            return failure();

        auto elemType = cast<RankedTensorType>(srcOp.getResultTypes()[0]).getElementType();

        // Build inputs and indexing maps for the replacement.
        // When the exponent was a tensor input, drop it so the result is a clean
        // single-input linalg.generic (avoids unused payload arguments).
        SmallVector<Value> newInputs;
        SmallVector<AffineMap> newMaps;
        auto maps = srcOp.getIndexingMapsArray();

        if (expMatch.isInputTensor) {
            for (unsigned i = 0; i < srcOp.getNumDpsInputs(); i++) {
                if (i != expMatch.inputIdx) {
                    newInputs.push_back(srcOp.getInputs()[i]);
                    newMaps.push_back(maps[i]);
                }
            }
            // Append output indexing maps
            for (unsigned i = srcOp.getNumDpsInputs(); i < maps.size(); i++) {
                newMaps.push_back(maps[i]);
            }
        }
        else {
            newInputs = SmallVector<Value>(srcOp.getInputs());
            newMaps = SmallVector<AffineMap>(maps);
        }

        // Replace pow(x, 2) with mul(x, x)
        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            srcOp, srcOp.getResultTypes(), newInputs, srcOp.getOutputs(), newMaps,
            srcOp.getIteratorTypesArray(),
            [&](OpBuilder &b, Location l, ValueRange args) {
                Value val;
                if (mlir::isa<FloatType>(elemType)) {
                    val = arith::MulFOp::create(b, l, args[0], args[0]);
                }
                else if (mlir::isa<IntegerType>(elemType)) {
                    val = arith::MulIOp::create(b, l, args[0], args[0]);
                }
                else {
                    // fallback: yield input as-is (should not happen)
                    val = args[0];
                }
                linalg::YieldOp::create(b, l, val);
            }
        );
        return success();
    }
};

class BfloatDivfPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    // using OpRewritePattern::OpRewritePattern;
    BfloatDivfPattern(MLIRContext *context)
        : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/0) {
        setDebugName("BfloatDivfPattern");
    }

    LogicalResult maybeMatchConstDiv(linalg::GenericOp srcOp, PatternRewriter &rewriter) const {

        auto yield = dyn_cast<linalg::YieldOp>(srcOp.getBody()->getTerminator());
        if (!yield || yield.getNumOperands() != 1)
            return rewriter.notifyMatchFailure(srcOp, "div requires one output");

        auto div = yield.getOperand(0).getDefiningOp<arith::DivFOp>();
        if (!div)
            return rewriter.notifyMatchFailure(srcOp, "div should end in div");

        auto constant = div.getRhs().getDefiningOp<arith::ConstantOp>();
        if (!isa<BlockArgument>(div.getLhs()) || !constant)
            return rewriter.notifyMatchFailure(
                srcOp, "Constant div matching requires non-const numerator"
            );

        auto bf16 = cast<RankedTensorType>(srcOp.getType(0)).getElementType();
        float denom = cast<FloatAttr>(constant.getValueAttr()).getValue().convertToDouble();
        // denom = 0 is accepted by the hardware and results in inf output
        auto recip = arith::ConstantOp::create(
            rewriter, srcOp.getLoc(), rewriter.getFloatAttr(bf16, 1 / denom)
        );

        rewriter.replaceOp(
            srcOp, linalg::GenericOp::create(
                       rewriter, srcOp.getLoc(), TypeRange{srcOp.getOutputs()[0].getType()},
                       srcOp.getInputs(), srcOp.getOutputs(), srcOp.getIndexingMapsArray(),
                       srcOp.getIteratorTypesArray(),
                       [&](OpBuilder &b, Location l, ValueRange args) {
                           linalg::YieldOp::create(
                               rewriter, l,
                               ValueRange{arith::MulFOp::create(rewriter, l, args[0], recip)}
                           );
                       }
                   )
        );
        return success();
    }

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {

        if (!cast<RankedTensorType>(srcOp.getType(0)).getElementType().isBF16())
            return rewriter.notifyMatchFailure(srcOp, "Expected bf16 for divf matching");

        if (!isa_and_nonnull<arith::DivFOp>(getElementwiseBinaryOp(srcOp)))
            return maybeMatchConstDiv(srcOp, rewriter);

        auto [isForced, forced] = setTargetExecutorIfForced(srcOp, rewriter, "div");
        if (isForced) {
            return forced;
        }

        auto numerator = srcOp.getOperand(0);
        auto denominator = srcOp.getOperand(1);
        auto output = srcOp.getOutputs()[0];

        SmallVector<Value> inputs = srcOp.getInputs();
        if (succeeded(broadcastInputs(srcOp, inputs, rewriter))) {
            LLVM_DEBUG(
                llvm::dbgs() << "Broadcasted inputs for divf pattern in op: " << srcOp << "\n"
            );
            numerator = inputs[0];
            denominator = inputs[1];
        }

        SmallVector<AffineMap> maps;
        maps.append(
            srcOp.getNumOperands(),
            AffineMap::getMultiDimIdentityMap(
                cast<RankedTensorType>(output.getType()).getRank(), rewriter.getContext()
            )
        );
        auto recip =
            linalg::ReciprocalOp::create(rewriter, srcOp.getLoc(), denominator, srcOp.getOutputs())
                .getResult(0);

        rewriter.replaceOp(
            srcOp,
            linalg::GenericOp::create(
                rewriter, srcOp.getLoc(), TypeRange{output.getType()}, ValueRange{numerator, recip},
                ValueRange{output}, maps, srcOp.getIteratorTypesArray(),
                [&](OpBuilder &b, Location l, ValueRange args) {
                    linalg::YieldOp::create(
                        rewriter, l,
                        ValueRange{arith::MulFOp::create(rewriter, l, args[0], args[1])}
                    );
                }
            )
        );
        return success();
    }
};

static llvm::cl::opt<bool> clEnableTorqReciprocalInf(
    "torq-enable-reciprocal-inf",
    llvm::cl::desc("enable torq support for inf input and output for reciprocal operation"),
    llvm::cl::init(false)
);

struct BfloatReciprocalPattern : public OpRewritePattern<linalg::ReciprocalOp> {
    BfloatReciprocalPattern(MLIRContext *context)
        : OpRewritePattern<linalg::ReciprocalOp>(context, /*benefit=*/0) {
        setDebugName("BfloatReciprocalPattern");
    }
    LogicalResult
    matchAndRewrite(linalg::ReciprocalOp op, PatternRewriter &rewriter) const override {

        // The default (non-inf) bf16 reciprocal is lowered directly from
        // linalg.reciprocal by ReciprocalOpPattern in LinalgToTorqHL; this
        // decomposition only handles the --torq-enable-reciprocal-inf variant.
        if (!clEnableTorqReciprocalInf) {
            return rewriter.notifyMatchFailure(
                op, "non-inf reciprocal is lowered in LinalgToTorqHL"
            );
        }

        // checks
        if (op.getInputs().size() != 1) {
            return rewriter.notifyMatchFailure(op, "Expected exactly one result");
        }
        if (op.getOutputs().size() != 1) {
            return rewriter.notifyMatchFailure(op, "Expected exactly one input");
        }
        if (!cast<RankedTensorType>(op.getType(0)).getElementType().isBF16()) {
            return rewriter.notifyMatchFailure(op, "Expected bf16 output");
        }
        auto [isForced, forced] = setTargetExecutorIfForced(op, rewriter, "reciprocal");
        if (isForced) {
            return forced;
        }

        // Matched!  Define some useful values.
        auto loc = op.getLoc();
        auto ctx = rewriter.getContext();
        auto bfTensorType = cast<RankedTensorType>(op.getType(0));
        auto shape = bfTensorType.getShape();
        auto rank = (size_t)bfTensorType.getRank();
        auto i1 = rewriter.getIntegerType(1);
        auto i8 = rewriter.getIntegerType(8);
        auto i16 = rewriter.getIntegerType(16);
        auto tType = [&](Type t) { return RankedTensorType::get(shape, t); };

        auto broadcast = [&](Value v, Type t, auto func) {
            return linalg::GenericOp::create(
                       rewriter, loc, TypeRange{tType(t)}, ValueRange{v},
                       ValueRange{tensor::EmptyOp::create(rewriter, loc, shape, t)},
                       SmallVector<AffineMap>(2, AffineMap::getMultiDimIdentityMap(rank, ctx)),
                       SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                       [&](OpBuilder &b, Location l, ValueRange args) {
                           linalg::YieldOp::create(rewriter, l, ValueRange{func(b, l, args)});
                       }
            ).getResult(0);
        };
        auto i16Const = [&](int constant) {
            return arith::ConstantOp::create(rewriter, loc, rewriter.getIntegerAttr(i16, constant));
        };
        auto andi = [&](int constant, Value val) {
            auto constVal = i16Const(constant);
            return broadcast(val, i16, [&](OpBuilder &b, Location l, ValueRange args) {
                return arith::AndIOp::create(b, loc, constVal, args[0]);
            });
        };
        auto mul = [&](int constant, Value val) {
            auto constVal = i16Const(constant);
            return broadcast(val, i16, [&](OpBuilder &b, Location l, ValueRange args) {
                return arith::MulIOp::create(b, loc, constVal, args[0]);
            });
        };
        auto sub = [&](int constant, Value val) {
            auto constVal = i16Const(constant);
            return broadcast(val, i16, [&](OpBuilder &b, Location l, ValueRange args) {
                return arith::SubIOp::create(b, loc, constVal, args[0]);
            });
        };
        auto cmp = [&](int constant, Value val, arith::CmpIPredicate pred) {
            auto constVal = i16Const(constant);
            auto boolVal = broadcast(val, i1, [&](OpBuilder &b, Location l, ValueRange args) {
                return arith::CmpIOp::create(b, l, pred, args[0], constVal);
            });
            return broadcast(boolVal, i16, [&](OpBuilder &b, Location l, ValueRange args) {
                return arith::ExtUIOp::create(b, l, i16, args[0]);
            });
        };

        // Work on the sign-stripped bit pattern: the sign bit is recorded
        // separately and re-added at the end.
        auto rawX = tensor::BitcastOp::create(rewriter, loc, tType(i16), op.getInputs()[0]);

        // record sign bit
        auto xSign = andi(0b1000000000000000, rawX);

        // remove sign bit for following logic
        auto x = andi(0b0111111111111111, rawX);

        // subnormal numbers are exactly the ones that map to
        // infinity.  Check which ones should map there.
        auto isSubnormal = cmp(0b0000000010000000, x, arith::CmpIPredicate::ult);

        // "big" means numbers that map to zero (we flush subnormals to zero).
        auto isNotBig = cmp(0b0111111010000000, x, arith::CmpIPredicate::ule);

        // magic LUT values (see getBfloatReciprocalLutData for the derivation)
        auto lutData = getBfloatReciprocalLutData();
        auto computedExpo = sub(0b0111111010000000, x);

        // The original (slow) linalg solution: separate trunci/exti tensor
        // ops around an i8 table lookup.  Kept for the
        // --torq-enable-reciprocal-inf fallback, where inf correctness
        // matters more than the two extra slices.
        // create LUT
        auto lut = arith::ConstantOp::create(
            rewriter, loc,
            DenseIntElementsAttr::get(RankedTensorType::get(256, i8), ArrayRef<int8_t>(lutData))
        );

        // use mantissa as index to LUT
        auto xMant8 = arith::TruncIOp::create(rewriter, loc, tType(i8), x);
        auto outputTens = tensor::EmptyOp::create(rewriter, loc, tType(i8), ValueRange{});
        auto indexOffset = arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(128));
        auto lutVal =
            linalg::GenericOp::create(
                rewriter, loc, TypeRange{tType(i8)}, ValueRange{xMant8}, ValueRange{outputTens},
                SmallVector<AffineMap>(2, AffineMap::getMultiDimIdentityMap(rank, ctx)),
                SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                [&](OpBuilder &b, Location loc, ValueRange args) {
                    auto baseIndex =
                        arith::IndexCastOp::create(b, loc, rewriter.getIndexType(), args[0]);
                    auto index = arith::AddIOp::create(b, loc, baseIndex, indexOffset);
                    auto extracted = tensor::ExtractOp::create(b, loc, lut, ValueRange{index});
                    linalg::YieldOp::create(b, loc, ValueRange{extracted});
                }
            ).getResult(0);

        auto lutVal16 = arith::ExtUIOp::create(rewriter, loc, tType(i16), lutVal);

        // combine computed exponent and mantissa
        auto computed = arith::AddIOp::create(rewriter, loc, lutVal16, computedExpo);

        // There are a few possible inputs (big inputs) that must also
        // be sqashed to zero.
        auto computed2 = arith::MulIOp::create(rewriter, loc, computed, isNotBig);

        // conversely, we need a way to ensure all subnormals map to
        // infinity.
        auto maybeInf = mul(0b0111111110000000, isSubnormal);

        // We want to combine our computed maybeInf and our current
        // computed using bfloat addition.  Bitcast it real quick,
        // add, and bitcast back.
        auto computedBfloat = tensor::BitcastOp::create(rewriter, loc, bfTensorType, computed2);
        auto maybeInfBfloat = tensor::BitcastOp::create(rewriter, loc, bfTensorType, maybeInf);
        auto realComputedBfloat =
            arith::AddFOp::create(rewriter, loc, computedBfloat, maybeInfBfloat);
        auto realComputed =
            tensor::BitcastOp::create(rewriter, loc, tType(i16), realComputedBfloat);

        // add back our sign bit we saved earlier
        computed = arith::AddIOp::create(rewriter, loc, xSign, realComputed);

        // bitcast final value back to bfloat16
        auto bfBitcast = tensor::BitcastOp::create(rewriter, loc, bfTensorType, computed);

        // done.
        rewriter.replaceOp(op, bfBitcast);
        return success();
    }
};

class BfloatGenericErfPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    // using OpRewritePattern::OpRewritePattern;
    BfloatGenericErfPattern(MLIRContext *context)
        : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/0) {
        setDebugName("BfloatGenericErfPattern");
    }

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {
        if (!cast<RankedTensorType>(srcOp.getType(0)).getElementType().isBF16() ||
            !isa_and_nonnull<math::ErfOp>(getElementwiseUnaryOp(srcOp))) {
            return rewriter.notifyMatchFailure(srcOp, "Expected bf16 erf");
        }
        rewriter.replaceOp(srcOp, math::ErfOp::create(rewriter, srcOp.getLoc(), srcOp.getInputs()));
        return success();
    }
};

struct BfloatErfPattern : public OpRewritePattern<math::ErfOp> {
    BfloatErfPattern(MLIRContext *context) : OpRewritePattern<math::ErfOp>(context, /*benefit=*/0) {
        setDebugName("BfloatErfPattern");
    }
    LogicalResult matchAndRewrite(math::ErfOp op, PatternRewriter &rewriter) const override {

        // checks
        auto bfTensorType = dyn_cast<RankedTensorType>(op.getType());
        if (!bfTensorType || !bfTensorType.getElementType().isBF16()) {
            return rewriter.notifyMatchFailure(op, "Expected bf16 output");
        }
        auto [isForced, forced] = setTargetExecutorIfForced(op, rewriter, "erf");
        if (isForced) {
            return forced;
        }

        // useful meta-values.
        auto loc = op.getLoc();
        auto ctx = rewriter.getContext();
        auto bf16 = bfTensorType.getElementType();
        auto rank = (size_t)bfTensorType.getRank();

        // helper functions.
        auto emptyLike = [&](Value v) {
            return tensor::EmptyOp::create(rewriter, loc, v.getType(), ValueRange{});
        };
        auto broadcast = [&](Value v, auto func) {
            return linalg::GenericOp::create(
                       rewriter, loc, TypeRange{v.getType()}, ValueRange{v},
                       ValueRange{emptyLike(v)},
                       SmallVector<AffineMap>(2, AffineMap::getMultiDimIdentityMap(rank, ctx)),
                       SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel), func
            )
                .getResult(0);
        };
        // x and y are tensors and z is a scalar constant.
        auto fma = [&](Value x, Value y, Value z) {
            auto xy = arith::MulFOp::create(rewriter, loc, x, y);
            return broadcast(xy, [&](OpBuilder &b, Location l, ValueRange args) {
                linalg::YieldOp::create(b, l, ValueRange{arith::AddFOp::create(b, l, args[0], z)});
            });
        };
        // y is a tensor and x and z are scalar constants.
        auto fmaConstant = [&](Value x, Value y, Value z) {
            auto xy = broadcast(y, [&](OpBuilder &b, Location l, ValueRange args) {
                linalg::YieldOp::create(b, l, ValueRange{arith::MulFOp::create(b, l, x, args[0])});
            });
            return broadcast(xy, [&](OpBuilder &b, Location l, ValueRange args) {
                linalg::YieldOp::create(b, l, ValueRange{arith::AddFOp::create(b, l, args[0], z)});
            });
        };
        // tensor x and contant scalar y
        auto mul = [&](Value x, Value y) {
            return broadcast(x, [&](OpBuilder &b, Location l, ValueRange args) {
                linalg::YieldOp::create(b, l, ValueRange{arith::MulFOp::create(b, l, args[0], y)});
            });
        };
        auto fpConst = [&](float fp) {
            return arith::ConstantOp::create(rewriter, loc, rewriter.getFloatAttr(bf16, fp));
        };

        // set some constants
        float pi = 3.141592653589793;
        auto scale = fpConst(2 / sqrt(pi));
        auto one = fpConst(1.0);
        auto mone = fpConst(-1.0);
        // Since odd terms have negative sign on leading term, you
        // would need to clip on input instead of output because the
        // approximation will switch signs at large values. The
        // maximum discrepancy in bits (i.e. 0b100 diff is 8 bits) for
        // number of terms is:
        /* 2 terms -> 21 bits */
        /* 4 terms -> 10 bits */
        /* 6 terms ->  5 bits */
        // FIXME: These values are obtained from taylor expansion
        // instead of Remez Algorithm to enable work on lowering now
        // to meet deadlines.  They should be updated to Remez
        // Algorithm values in the future.
        auto c0 = one; // fpConst(1.0);
        auto c1 = fpConst(-1.0 / 3.0);
        auto c2 = fpConst(1.0 / 10.0);
        auto c3 = fpConst(-1.0 / 42.0);
        auto c4 = fpConst(1.0 / 216.0);
        auto c5 = fpConst(-1.0 / 1320.0);
        auto c6 = fpConst(1.0 / 9360.0);

        // run the algorithm
        auto x = op.getOperand();
        auto x2 = arith::MulFOp::create(rewriter, loc, x, x);
        auto terms =
            fma(fma(fma(fma(fma(fmaConstant(c6, x2, c5), x2, c4), x2, c3), x2, c2), x2, c1), x2,
                c0);
        auto unclamped = arith::MulFOp::create(rewriter, loc, terms, mul(x, scale));
        auto outAlloc = tensor::EmptyOp::create(rewriter, loc, x.getType(), ValueRange{});
        auto erf =
            linalg::GenericOp::create(
                rewriter, loc, TypeRange{outAlloc.getType()}, ValueRange{unclamped},
                ValueRange{outAlloc},
                SmallVector<AffineMap>(2, AffineMap::getMultiDimIdentityMap(rank, ctx)),
                SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                [&](OpBuilder &b, Location loc, ValueRange args) {
                    auto tooLarge =
                        arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT, one, args[0]);
                    auto clamped1 = arith::SelectOp::create(b, loc, tooLarge, one, args[0]);
                    auto tooSmall =
                        arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGT, mone, clamped1);
                    auto clamped2 = arith::SelectOp::create(b, loc, tooSmall, mone, clamped1);
                    linalg::YieldOp::create(b, loc, ValueRange{clamped2});
                }
            ).getResult(0);

        rewriter.replaceOp(op, erf);
        return success();
    }
};

// A pattern to lower linalg.quantized_batch_matmul to the following sequence:
// 1) linalg.add i8, i8 -> i16 (to handle zero point)
// 2) linalg.matmul i16, i16 -> i32
struct QuantizedBatchMatmulPattern : public OpRewritePattern<linalg::QuantizedBatchMatmulOp> {
    QuantizedBatchMatmulPattern(MLIRContext *context)
        : OpRewritePattern<linalg::QuantizedBatchMatmulOp>(context, /*benefit=*/0) {
        setDebugName("QuantizedBatchMatmulPattern");
    }

    // Add a dynamic scalar (0-D tensor) to a tensor<i16> using linalg.generic.
    Value addScalar0DWithGeneric(
        PatternRewriter &rewriter, Location loc, Value tensorI16, Value negZpI16
    ) const {
        auto tTy = cast<RankedTensorType>(tensorI16.getType());
        Type i16 = tTy.getElementType();

        // Wrap the scalar as tensor<i16> (rank-0).
        RankedTensorType scalarTy = RankedTensorType::get({}, i16);
        auto empty = tensor::EmptyOp::create(rewriter, loc, scalarTy, ValueRange{});
        Value scalar0D = tensor::InsertOp::create(rewriter, loc, negZpI16, empty, ValueRange{});

        // FIXME: we don't support tensor-scalar arith ops when scalar is non constant
        // so we create a tensor for the scalar with the same shape as the input tensor.
        // This is not efficient but should be ok for now as the scalar is expected to be
        // a zero point.
        Value cEmpty = tensor::EmptyOp::create(rewriter, loc, tTy.getShape(), i16);

        SmallVector<int64_t> dim(tTy.getRank());
        for (int i = 0; i < tTy.getRank(); i++) {
            dim[i] = i;
        }
        Value cInit =
            linalg::BroadcastOp::create(rewriter, loc, scalar0D, cEmpty, dim).getResults()[0];

        // Identity for the tensor, scalar map ()->() for the 0-D input, identity for the output.
        AffineMap id = rewriter.getMultiDimIdentityMap(tTy.getRank());
        SmallVector<utils::IteratorType> iters(tTy.getRank(), utils::IteratorType::parallel);

        Value out = tensor::EmptyOp::create(rewriter, loc, tTy.getShape(), tTy.getElementType());
        auto g = linalg::GenericOp::create(
            rewriter, loc, TypeRange{tTy}, ValueRange{tensorI16, cInit}, ValueRange{out},
            ArrayRef<AffineMap>{id, id, id}, iters,
            [&](OpBuilder &b, Location l, ValueRange args) {
                // TODO: double-check if zeropoint must be substracted or added using llvm-cpu with
                // a given input matrix
                Value sum = arith::SubIOp::create(b, l, args[0], args[1]);
                linalg::YieldOp::create(b, l, sum);
            }
        );
        return g.getResult(0);
    }

    LogicalResult
    matchAndRewrite(linalg::QuantizedBatchMatmulOp op, PatternRewriter &rewriter) const override {

        Location loc = op.getLoc();

        // Expect: A(i8), B(i8), zpL(any int), zpR(any int); output init i32 tensor.
        Value aI8 = op.getDpsInputOperand(0)->get();
        Value bI8 = op.getDpsInputOperand(1)->get();
        Value zpLRaw = op->getOperand(2);
        Value zpRRaw = op->getOperand(3);
        Value outInit = op.getDpsInitOperand(0)->get();

        auto aTy = dyn_cast<RankedTensorType>(aI8.getType());
        auto bTy = dyn_cast<RankedTensorType>(bI8.getType());
        auto outTy = dyn_cast<RankedTensorType>(outInit.getType());
        if (!aTy || !bTy || !outTy)
            return rewriter.notifyMatchFailure(op, "expected ranked tensors");
        if (!aTy.getElementType().isInteger(8) || !bTy.getElementType().isInteger(8) ||
            !outTy.getElementType().isInteger(32))
            return rewriter.notifyMatchFailure(op, "expected A/B i8 and out i32");

        Type i16 = rewriter.getIntegerType(16);
        Value aI16 =
            arith::ExtSIOp::create(rewriter, loc, RankedTensorType::get(aTy.getShape(), i16), aI8);
        Value bI16 =
            arith::ExtSIOp::create(rewriter, loc, RankedTensorType::get(bTy.getShape(), i16), bI8);

        auto zpL16 = arith::TruncIOp::create(rewriter, loc, i16, zpLRaw); // i8/i16 -> i16
        auto zpR16 = arith::TruncIOp::create(rewriter, loc, i16, zpRRaw); // i8/i16 -> i16

        Value aAdj = addScalar0DWithGeneric(rewriter, loc, aI16, zpL16);
        Value bAdj = addScalar0DWithGeneric(rewriter, loc, bI16, zpR16);

        // Zero i32 accumulator and (batch_)matmul i16×i16 → i32.
        Value cEmpty =
            tensor::EmptyOp::create(rewriter, loc, outTy.getShape(), outTy.getElementType());
        Value z32 = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
        Value cInit =
            linalg::FillOp::create(rewriter, loc, ValueRange{z32}, ValueRange{cEmpty}).result();

        Value result;
        if (outTy.getRank() == 3) {
            result = linalg::BatchMatmulOp::create(
                         rewriter, loc, TypeRange{outTy}, ValueRange{aAdj, bAdj}, ValueRange{cInit}
            )
                         .getResult(0);
        }
        else {
            result = linalg::MatmulOp::create(
                         rewriter, loc, TypeRange{outTy}, ValueRange{aAdj, bAdj}, ValueRange{cInit}
            )
                         .getResult(0);
        }

        rewriter.replaceOp(op, result);
        return success();
    }
};

// Decompose bf16 sqrt(x) into x * rsqrt(x).
// The resulting rsqrt linalg.generic is further decomposed by BfloatRsqrtPattern.
class BfloatSqrtPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    BfloatSqrtPattern(MLIRContext *context)
        : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/1) {
        setDebugName("BfloatSqrtPattern");
    }

    LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
        if (!cast<RankedTensorType>(op.getType(0)).getElementType().isBF16() ||
            !isa_and_nonnull<math::SqrtOp>(getElementwiseUnaryOp(op))) {
            return rewriter.notifyMatchFailure(op, "Expected bf16 sqrt");
        }
        auto [isForced, forced] = setTargetExecutorIfForced(op, rewriter, "sqrt");
        if (isForced) {
            return forced;
        }

        auto loc = op.getLoc();
        auto input = op.getInputs()[0];
        auto resultType = cast<RankedTensorType>(op.getType(0));
        auto rank = resultType.getRank();
        auto ctx = rewriter.getContext();

        // Create linalg.generic with math.rsqrt in body
        auto rsqrt = linalg::GenericOp::create(
            rewriter, loc, TypeRange{resultType}, ValueRange{input},
            ValueRange{tensor::EmptyOp::create(rewriter, loc, resultType, ValueRange{})},
            SmallVector<AffineMap>(2, AffineMap::getMultiDimIdentityMap(rank, ctx)),
            SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
            [&](OpBuilder &b, Location l, ValueRange args) {
                auto rsqrtVal = math::RsqrtOp::create(b, l, args[0]);
                linalg::YieldOp::create(b, l, ValueRange{rsqrtVal});
            }
        );

        auto result = arith::MulFOp::create(rewriter, loc, input, rsqrt.getResult(0));
        rewriter.replaceOp(op, result);
        return success();
    }
};

class BfloatRsqrtPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    // using OpRewritePattern::OpRewritePattern;
    BfloatRsqrtPattern(MLIRContext *context)
        : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/0) {
        setDebugName("BfloatRsqrtPattern");
    }

    LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
        if (!cast<RankedTensorType>(op.getType(0)).getElementType().isBF16() ||
            !isa_and_nonnull<math::RsqrtOp>(getElementwiseUnaryOp(op))) {
            return rewriter.notifyMatchFailure(op, "Expected bf16 rsqrt");
        }
        auto [isForced, forced] = setTargetExecutorIfForced(op, rewriter, "rsqrt");
        if (isForced) {
            return forced;
        }

        auto loc = op.getLoc();
        auto bfTensorType = cast<RankedTensorType>(op.getType(0));
        auto bf16 = bfTensorType.getElementType();
        auto i16 = rewriter.getIntegerType(16);
        auto i16TensorType = RankedTensorType::get(bfTensorType.getShape(), i16);
        auto rank = bfTensorType.getRank();
        auto ctx = rewriter.getContext();

        auto broadcast = [&](Value v, RankedTensorType t, auto func) {
            return linalg::GenericOp::create(
                       rewriter, loc, TypeRange{t}, ValueRange{v},
                       ValueRange{tensor::EmptyOp::create(rewriter, loc, t, ValueRange{})},
                       SmallVector<AffineMap>(2, AffineMap::getMultiDimIdentityMap(rank, ctx)),
                       SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                       [&](OpBuilder &b, Location l, ValueRange args) {
                           linalg::YieldOp::create(b, l, ValueRange{func(b, l, args)});
                       }
            ).getResult(0);
        };

        // following https://en.wikipedia.org/wiki/Fast_inverse_square_root
        auto number = op.getInputs()[0];
        const int16_t rsqrtMagic = 0x5f37;
        auto magic =
            arith::ConstantOp::create(rewriter, loc, rewriter.getIntegerAttr(i16, rsqrtMagic));
        auto threeHalfs =
            arith::ConstantOp::create(rewriter, loc, rewriter.getFloatAttr(bf16, 1.5));
        auto oneHalf = arith::ConstantOp::create(rewriter, loc, rewriter.getFloatAttr(bf16, 0.5));
        auto x2 = broadcast(number, bfTensorType, [&](OpBuilder &b, Location l, ValueRange args) {
            return arith::MulFOp::create(b, l, oneHalf, args[0]);
        });
        auto i = tensor::BitcastOp::create(rewriter, loc, i16TensorType, number);
        auto one = arith::ConstantOp::create(rewriter, loc, rewriter.getIntegerAttr(i16, 1));
        auto i2 = broadcast(
            broadcast(
                i, i16TensorType,
                [&](OpBuilder &b, Location l, ValueRange args) {
                    return arith::ShRSIOp::create(b, loc, args[0], one);
                }
            ),
            i16TensorType,
            [&](OpBuilder &b, Location l, ValueRange args) {
                return arith::SubIOp::create(b, loc, magic, args[0]);
            }
        );
        auto y = tensor::BitcastOp::create(rewriter, loc, bfTensorType, i2);
        auto y2 = arith::MulFOp::create(
            rewriter, loc, y,
            broadcast(
                arith::MulFOp::create(
                    rewriter, loc, arith::MulFOp::create(rewriter, loc, x2, y), y
                ),
                bfTensorType,
                [&](OpBuilder &b, Location l, ValueRange args) {
                    return arith::SubFOp::create(rewriter, loc, threeHalfs, args[0]);
                }
            )
        );
        rewriter.replaceOp(op, y2);
        return success();
    }
};

// Pattern to fold NHWC DepthToSpace (expand_shape → transpose → collapse_shape)
// into a single linalg.generic with computed indexing map.
//
// Matches:
//   %expanded = tensor.expand_shape %input [[0], [1], [2], [3, 4, 5]]
//                 : tensor<NxHxWxC> into tensor<NxHxWx(bH)x(bW)x(C_out)>
//   %transposed = linalg.transpose ins(%expanded) permutation = [0, 1, 3, 2, 4, 5]
//                 : tensor<NxHxWx(bH)x(bW)x(C_out)> into tensor<NxHx(bH)xWx(bW)x(C_out)>
//   %collapsed = tensor.collapse_shape %transposed [[0], [1, 2], [3, 4], [5]]
//                 : tensor<NxHx(bH)xWx(bW)x(C_out)> into tensor<Nx(H')x(W')x(C_out)>
//
// Replaces with:
//   transpose NHWC -> NCHW
//   %result = linalg.generic with computed indexing map (NCHW)
//   transpose NCHW -> NHWC
//
struct FoldNhwcDepthToSpacePattern : public OpRewritePattern<tensor::CollapseShapeOp> {
    FoldNhwcDepthToSpacePattern(MLIRContext *context)
        : OpRewritePattern<tensor::CollapseShapeOp>(context, /*benefit=*/10) {
        setDebugName("FoldNhwcDepthToSpacePattern");
    }

    LogicalResult
    matchAndRewrite(tensor::CollapseShapeOp collapseOp, PatternRewriter &rewriter) const override {
        // Check collapse pattern: [[0], [1, 2], [3, 4], [5]]
        auto collapseIndices = collapseOp.getReassociationIndices();
        if (collapseIndices.size() != 4 || collapseIndices[0] != ArrayRef<int64_t>{0} ||
            collapseIndices[1] != ArrayRef<int64_t>{1, 2} ||
            collapseIndices[2] != ArrayRef<int64_t>{3, 4} ||
            collapseIndices[3] != ArrayRef<int64_t>{5}) {
            return rewriter.notifyMatchFailure(
                collapseOp, "Not NHWC DepthToSpace collapse pattern"
            );
        }

        // Get transpose op
        auto transposeOp = collapseOp.getSrc().getDefiningOp<linalg::TransposeOp>();
        if (!transposeOp)
            return rewriter.notifyMatchFailure(collapseOp, "Input not from transpose");

        // Check transpose permutation: [0, 1, 3, 2, 4, 5]
        auto perm = transposeOp.getPermutation();
        if (perm != ArrayRef<int64_t>{0, 1, 3, 2, 4, 5})
            return rewriter.notifyMatchFailure(collapseOp, "Not NHWC DepthToSpace transpose");

        // Get expand op
        auto expandOp = transposeOp.getOperand(0).getDefiningOp<tensor::ExpandShapeOp>();
        if (!expandOp)
            return rewriter.notifyMatchFailure(collapseOp, "Transpose input not from expand");

        // Check expand pattern: [[0], [1], [2], [3, 4, 5]]
        auto expandIndices = expandOp.getReassociationIndices();
        if (expandIndices.size() != 4 || expandIndices[0] != ArrayRef<int64_t>{0} ||
            expandIndices[1] != ArrayRef<int64_t>{1} || expandIndices[2] != ArrayRef<int64_t>{2} ||
            expandIndices[3] != ArrayRef<int64_t>{3, 4, 5}) {
            return rewriter.notifyMatchFailure(collapseOp, "Not NHWC DepthToSpace expand pattern");
        }

        // Get input and output types
        auto inputType = dyn_cast<RankedTensorType>(expandOp.getOperand(0).getType());
        auto outputType = dyn_cast<RankedTensorType>(collapseOp.getResult().getType());
        if (!inputType || !outputType || inputType.getRank() != 4 || outputType.getRank() != 4)
            return rewriter.notifyMatchFailure(collapseOp, "Expected 4D tensors");

        auto inputShape = inputType.getShape();   // [N, H, W, C] NHWC
        auto outputShape = outputType.getShape(); // [N, H', W', C_out] NHWC
        auto expandedShape = cast<RankedTensorType>(expandOp.getResult().getType()).getShape();
        // expandedShape = [N, H, W, bH, bW, C_out]

        int64_t N = inputShape[0];
        int64_t H = inputShape[1];
        int64_t W = inputShape[2];
        int64_t C = inputShape[3];

        int64_t bH = expandedShape[3];
        int64_t bW = expandedShape[4];
        int64_t C_out = expandedShape[5];

        int64_t H_out = outputShape[1];
        int64_t W_out = outputShape[2];

        // Verify relationships
        if (C != bH * bW * C_out || H_out != bH * H || W_out != bW * W) {
            return rewriter.notifyMatchFailure(collapseOp, "DepthToSpace dimension mismatch");
        }

        MLIRContext *ctx = rewriter.getContext();
        Location loc = collapseOp.getLoc();
        Type elemType = inputType.getElementType();

        // Step 1: Transpose input NHWC -> NCHW
        Value nchwInput =
            transposeValue(expandOp.getOperand(0), Permutation::nhwc2nchw(), loc, rewriter);

        // Step 2: Build indexing maps for NCHW DepthToSpace
        // Output: (d0, d1, d2, d3) -> (N, C_out, H', W')
        // Input: (d0, d1, d2, d3) -> (N, C, H, W) where:
        //   - N = d0
        //   - C = ((d2 mod bH) * bW + (d3 mod bW)) * C_out + d1
        //   - H = d2 floordiv bH
        //   - W = d3 floordiv bW

        auto d0 = getAffineDimExpr(0, ctx);
        auto d1 = getAffineDimExpr(1, ctx);
        auto d2 = getAffineDimExpr(2, ctx);
        auto d3 = getAffineDimExpr(3, ctx);

        auto blockH = d2 % bH;
        auto blockW = d3 % bW;
        auto blockIdx = blockH * bW + blockW;
        auto cIdx = blockIdx * C_out + d1;

        auto hIdx = d2.floorDiv(bH);
        auto wIdx = d3.floorDiv(bW);

        AffineMap inputMap = AffineMap::get(4, 0, {d0, cIdx, hIdx, wIdx}, ctx);
        AffineMap outputMap = AffineMap::getMultiDimIdentityMap(4, ctx);

        SmallVector<AffineMap> indexingMaps = {inputMap, outputMap};
        SmallVector<utils::IteratorType> iteratorTypes(4, utils::IteratorType::parallel);

        // Create output tensor in NCHW
        auto nchwOutputShape = SmallVector<int64_t>{N, C_out, H_out, W_out};
        Value nchwOutputInit = tensor::EmptyOp::create(rewriter, loc, nchwOutputShape, elemType);

        // Create linalg.generic for DepthToSpace in NCHW
        auto genericOp = linalg::GenericOp::create(
            rewriter, loc, TypeRange{nchwOutputInit.getType()}, ValueRange{nchwInput},
            ValueRange{nchwOutputInit}, indexingMaps, iteratorTypes,
            [&](OpBuilder &b, Location l, ValueRange args) {
                linalg::YieldOp::create(b, l, args[0]);
            }
        );

        // Step 3: Transpose output NCHW -> NHWC
        Value nhwcOutput =
            transposeValue(genericOp.getResult(0), Permutation::nchw2nhwc(), loc, rewriter);

        rewriter.replaceOp(collapseOp, nhwcOutput);

        // Clean up unused transpose and expand ops
        if (transposeOp.getResult().use_empty())
            rewriter.eraseOp(transposeOp);
        if (expandOp.getResult().use_empty())
            rewriter.eraseOp(expandOp);

        LLVM_DEBUG({
            llvm::dbgs() << "[FoldDepthToSpace] Folded NHWC DepthToSpace pattern:\n";
            llvm::dbgs() << "  Input: " << inputType << "\n";
            llvm::dbgs() << "  Output: " << outputType << "\n";
            llvm::dbgs() << "  Block size: " << bH << "x" << bW << "\n";
        });

        return success();
    }
};

void populateDecomposeLinalgOpsPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.insert<BfloatGenericErfPattern>(context);
    patterns.insert<BfloatErfPattern>(context);
    patterns.insert<BfloatDivfPattern>(context);
    patterns.insert<BfloatReciprocalPattern>(context);
    patterns.insert<QuantizedBatchMatmulPattern>(context);
    patterns.insert<PowToMulPattern>(context);
    patterns.insert<BfloatSqrtPattern>(context);
    patterns.insert<BfloatRsqrtPattern>(context);
    patterns.insert<CanonicalizeBF16CastComparePattern>(context);
    patterns.insert<FoldNhwcDepthToSpacePattern>(context);
}

} // namespace mlir::syna::torq
