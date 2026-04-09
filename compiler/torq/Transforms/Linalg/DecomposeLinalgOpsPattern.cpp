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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-decompose-linalg-ops-pattern"

using namespace std;

namespace mlir::syna::torq {

struct TensorBitcastPattern : public OpRewritePattern<tensor::BitcastOp> {
    TensorBitcastPattern(MLIRContext *context)
        : OpRewritePattern<tensor::BitcastOp>(context, /*benefit=*/0) {
        setDebugName("TensorBitcastPattern");
    }
    LogicalResult
    matchAndRewrite(tensor::BitcastOp bitcastOp, PatternRewriter &rewriter) const override {

        auto inputType = dyn_cast<RankedTensorType>(bitcastOp.getSource().getType());
        auto resultType = dyn_cast<RankedTensorType>(bitcastOp.getResult().getType());

        if (!inputType || !resultType) {
            return failure();
        }

        auto emptyOp = tensor::EmptyOp::create(
            rewriter, bitcastOp.getLoc(), resultType.getShape(), resultType.getElementType()
        );

        size_t rank = inputType.getRank();

        SmallVector<AffineMap> maps{
            2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext())
        };
        SmallVector<utils::IteratorType> iteratorTypes{rank, utils::IteratorType::parallel};

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            bitcastOp, resultType, ValueRange{bitcastOp.getSource()}, ValueRange{emptyOp}, maps,
            iteratorTypes,
            [&](OpBuilder &nestedBuilder, Location loc, ValueRange args) {
                auto castOp = arith::BitcastOp::create(
                    nestedBuilder, loc, resultType.getElementType(), args[0]
                );
                linalg::YieldOp::create(nestedBuilder, loc, ValueRange{castOp});
            }
        );

        return success();
    }
};

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

// Rewrites linalg.generic ops containing math.powf(x, cst) where cst is a constant 2
// (for bf16, i8, i16, i32 element types) into a linalg.generic with a multiplication
// (mul) of the input with itself. This is useful for lowering pow(x, 2) to mul(x, x)
// until the target dialect does not support powf.
class PowToMulPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    PowToMulPattern(MLIRContext *context)
        : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/0) {
        setDebugName("PowToMulPattern");
    }

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {
        // Only match elementwise binary ops
        auto powOp =
            dyn_cast_or_null<math::PowFOp>(getElementwiseBinaryOp(srcOp, /*allowConstant=*/true));
        if (!powOp) {
            return rewriter.notifyMatchFailure(srcOp, "Not a math.powf op");
        }
        auto rhs = powOp.getRhs();
        Attribute rhsAttr;

        // Only support constant RHS: check for arith.constant
        if (auto cOp = rhs.getDefiningOp<arith::ConstantOp>()) {
            rhsAttr = cOp.getValue();
        }
        else {
            return rewriter.notifyMatchFailure(srcOp, "RHS is not a constant");
        }

        Type rhsType = rhs.getType();
        if (rhsType.isF32()) {
            return rewriter.notifyMatchFailure(srcOp, "Type f32 not supported");
        }

        bool isTwo = false;
        if (rhsType.isBF16()) {
            if (auto floatAttr = mlir::dyn_cast<FloatAttr>(rhsAttr)) {
                isTwo = (floatAttr.getValueAsDouble() == 2.0);
            }
        }
        else if (rhsType.isInteger(8) || rhsType.isInteger(16) || rhsType.isInteger(32)) {
            if (auto intAttr = mlir::dyn_cast<IntegerAttr>(rhsAttr)) {
                isTwo = (intAttr.getInt() == 2);
            }
        }

        if (!isTwo) {
            return rewriter.notifyMatchFailure(srcOp, "RHS constant is not 2");
        }

        auto elemType = cast<RankedTensorType>(srcOp.getResultTypes()[0]).getElementType();

        // Replace powf(x, 2) with mul(x, x)
        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            srcOp, srcOp.getResultTypes(), srcOp.getInputs(), srcOp.getOutputs(),
            srcOp.getIndexingMapsArray(), srcOp.getIteratorTypesArray(),
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
        auto recip =
            linalg::ReciprocalOp::create(rewriter, srcOp.getLoc(), denominator, srcOp.getOutputs())
                .getResult(0);

        rewriter.replaceOp(
            srcOp,
            linalg::GenericOp::create(
                rewriter, srcOp.getLoc(), TypeRange{output.getType()}, ValueRange{numerator, recip},
                ValueRange{output}, srcOp.getIndexingMapsArray(), srcOp.getIteratorTypesArray(),
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

        // create variable names that survive the following if block
        Value rawX, xSign, x, isSubnormal, isNotBig;
        if (clEnableTorqReciprocalInf) {
            rawX = tensor::BitcastOp::create(rewriter, loc, tType(i16), op.getInputs()[0]);

            // record sign bit
            xSign = andi(0b1000000000000000, rawX);

            // remove sign bit for following logic
            x = andi(0b0111111111111111, rawX);

            // subnormal numbers are exactly the ones that map to
            // infinity.  Check which ones should map there.
            isSubnormal = cmp(0b0000000010000000, x, arith::CmpIPredicate::ult);

            // "big" means numbers that map to zero (we flush subnormals to zero).
            isNotBig = cmp(0b0111111010000000, x, arith::CmpIPredicate::ule);
        }
        else {
            // bitcast to i16
            x = tensor::BitcastOp::create(rewriter, loc, tType(i16), op.getInputs()[0]);
        }

        // Our magic LUT values!  See
        // scripts/torch/bfloat16_softmax.py to reproduce.
        std::vector<int8_t> lutData{
            // pad zeros for easier lowering
            -0x80, 0x7E, 0x7C, 0x7A, 0x78, 0x76, 0x75, 0x73, 0x71, 0x6F, 0x6D, 0x6C, 0x6A, 0x68,
            0x67, 0x65, 0x64, 0x62, 0x60, 0x5F, 0x5D, 0x5C, 0x5A, 0x59, 0x58, 0x56, 0x55, 0x53,
            0x52, 0x51, 0x4F, 0x4E, 0x4D, 0x4C, 0x4A, 0x49, 0x48, 0x47, 0x45, 0x44, 0x43, 0x42,
            0x41, 0x40, 0x3F, 0x3D, 0x3C, 0x3B, 0x3A, 0x39, 0x38, 0x37, 0x36, 0x35, 0x34, 0x33,
            0x32, 0x31, 0x30, 0x2F, 0x2E, 0x2D, 0x2C, 0x2C, 0x2B, 0x2A, 0x29, 0x28, 0x27, 0x26,
            0x25, 0x25, 0x24, 0x23, 0x22, 0x21, 0x21, 0x20, 0x1F, 0x1E, 0x1E, 0x1D, 0x1C, 0x1B,
            0x1B, 0x1A, 0x19, 0x18, 0x18, 0x17, 0x16, 0x16, 0x15, 0x14, 0x14, 0x13, 0x12, 0x12,
            0x11, 0x10, 0x10, 0x0F, 0x0E, 0x0E, 0x0D, 0x0D, 0x0C, 0x0B, 0x0B, 0x0A, 0x0A, 0x09,
            0x09, 0x08, 0x07, 0x07, 0x06, 0x06, 0x05, 0x05, 0x04, 0x04, 0x03, 0x03, 0x02, 0x02,
            0x01, 0x01,
            // actual values
            -0x80, 0x7E, 0x7C, 0x7A, 0x78, 0x76, 0x75, 0x73, 0x71, 0x6F, 0x6D, 0x6C, 0x6A, 0x68,
            0x67, 0x65, 0x64, 0x62, 0x60, 0x5F, 0x5D, 0x5C, 0x5A, 0x59, 0x58, 0x56, 0x55, 0x53,
            0x52, 0x51, 0x4F, 0x4E, 0x4D, 0x4C, 0x4A, 0x49, 0x48, 0x47, 0x45, 0x44, 0x43, 0x42,
            0x41, 0x40, 0x3F, 0x3D, 0x3C, 0x3B, 0x3A, 0x39, 0x38, 0x37, 0x36, 0x35, 0x34, 0x33,
            0x32, 0x31, 0x30, 0x2F, 0x2E, 0x2D, 0x2C, 0x2C, 0x2B, 0x2A, 0x29, 0x28, 0x27, 0x26,
            0x25, 0x25, 0x24, 0x23, 0x22, 0x21, 0x21, 0x20, 0x1F, 0x1E, 0x1E, 0x1D, 0x1C, 0x1B,
            0x1B, 0x1A, 0x19, 0x18, 0x18, 0x17, 0x16, 0x16, 0x15, 0x14, 0x14, 0x13, 0x12, 0x12,
            0x11, 0x10, 0x10, 0x0F, 0x0E, 0x0E, 0x0D, 0x0D, 0x0C, 0x0B, 0x0B, 0x0A, 0x0A, 0x09,
            0x09, 0x08, 0x07, 0x07, 0x06, 0x06, 0x05, 0x05, 0x04, 0x04, 0x03, 0x03, 0x02, 0x02,
            0x01, 0x01
        };

        // naively, we would subtract the exponent, and compute the
        // mantissa directly from the above lookup table.  However, to
        // avoid needing to & away the mantissa bits, leave them in
        // the exponent calculation and subtract that error from the
        // mantissa LUT.
        for (int i = 0; i < 256; i++) {
            lutData[i] = lutData[i] + (i % 128);
        }
        auto computedExpo = sub(0b0111111010000000, x);

        // create LUT
        auto lut = arith::ConstantOp::create(
            rewriter, loc,
            DenseIntElementsAttr::get(
                RankedTensorType::get(256, rewriter.getI8Type()), ArrayRef<int8_t>(lutData)
            )
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
        if (clEnableTorqReciprocalInf) {
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
        }

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
        if (!cast<RankedTensorType>(op.getType()).getElementType().isBF16()) {
            return rewriter.notifyMatchFailure(op, "Expected bf16 output");
        }
        auto [isForced, forced] = setTargetExecutorIfForced(op, rewriter, "erf");
        if (isForced) {
            return forced;
        }

        // useful meta-values.
        auto loc = op.getLoc();
        auto ctx = rewriter.getContext();
        auto bfTensorType = cast<RankedTensorType>(op.getType());
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
        // only works if x and y are tensor and z is a scalar constants.
        auto fma = [&](Value x, Value y, Value z) {
            auto xy = arith::MulFOp::create(rewriter, loc, x, y);
            return broadcast(xy, [&](OpBuilder &b, Location l, ValueRange args) {
                linalg::YieldOp::create(b, l, ValueRange{arith::AddFOp::create(b, l, args[0], z)});
            });
        };
        auto fmaConstant = [&](Value x, Value y, Value z) {
            auto xy = broadcast(y, [&](OpBuilder &b, Location l, ValueRange args) {
                linalg::YieldOp::create(b, l, ValueRange{arith::MulFOp::create(b, l, x, args[0])});
            });
            return broadcast(xy, [&](OpBuilder &b, Location l, ValueRange args) {
                linalg::YieldOp::create(b, l, ValueRange{arith::AddFOp::create(b, l, args[0], z)});
            });
        };
        // expects tenor x and contant scalar y
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
class BfloatGenericTanhPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    // using OpRewritePattern::OpRewritePattern;
    BfloatGenericTanhPattern(MLIRContext *context)
        : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/0) {
        setDebugName("BfloatGenericTanhPattern");
    }

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {
        if (!cast<RankedTensorType>(srcOp.getType(0)).getElementType().isBF16() ||
            !isa_and_nonnull<math::TanhOp>(getElementwiseUnaryOp(srcOp))) {
            return rewriter.notifyMatchFailure(srcOp, "Expected bf16 tanh");
        }
        rewriter.replaceOp(
            srcOp,
            linalg::TanhOp::create(rewriter, srcOp.getLoc(), srcOp.getInputs(), srcOp.getOutputs())
        );
        return success();
    }
};

struct BfloatTanhPattern : OpRewritePattern<linalg::TanhOp> {
    BfloatTanhPattern(MLIRContext *context)
        : OpRewritePattern<linalg::TanhOp>(context, /*benefit=*/0) {
        setDebugName("BfloatTanhPattern");
    }
    LogicalResult matchAndRewrite(linalg::TanhOp op, PatternRewriter &rewriter) const override {

        // checks
        if (!cast<RankedTensorType>(op.getType(0)).getElementType().isBF16()) {
            return rewriter.notifyMatchFailure(op, "Expected bf16 output");
        }
        auto [isForced, forced] = setTargetExecutorIfForced(op, rewriter, "tanh");
        if (isForced) {
            return forced;
        }

        auto ctx = rewriter.getContext();
        auto loc = op.getLoc();
        auto bf16TensType = cast<RankedTensorType>(op.getType(0));
        auto bf16 = bf16TensType.getElementType();
        auto rank = bf16TensType.getRank();

        auto emptyLike = [&](Value val) {
            return tensor::EmptyOp::create(rewriter, loc, val.getType(), ValueRange{});
        };
        auto typeOf = [&](Value v) { return v.getType(); };
        auto fpConst = [&](float fp) {
            return arith::ConstantOp::create(rewriter, loc, rewriter.getFloatAttr(bf16, fp));
        };
        // broadcast over one tensor
        auto broadcastOne = [&](Value v, auto func) {
            return linalg::GenericOp::create(
                       rewriter, loc, TypeRange{typeOf(v)}, ValueRange{v}, ValueRange{emptyLike(v)},
                       SmallVector<AffineMap>(2, AffineMap::getMultiDimIdentityMap(rank, ctx)),
                       SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel), func
            )
                .getResult(0);
        };
        // only works if x and y are tensor and z is a scalar constants.
        auto fmaTensor = [&](Value x, Value y, Value z) {
            auto xy = arith::MulFOp::create(rewriter, loc, x, y);
            return broadcastOne(xy, [&](OpBuilder &b, Location l, ValueRange args) {
                linalg::YieldOp::create(b, l, ValueRange{arith::AddFOp::create(b, l, args[0], z)});
            });
        };
        auto fmaConstant = [&](Value x, Value y, Value z) {
            auto xy = broadcastOne(y, [&](OpBuilder &b, Location l, ValueRange args) {
                linalg::YieldOp::create(b, l, ValueRange{arith::MulFOp::create(b, l, x, args[0])});
            });
            return broadcastOne(xy, [&](OpBuilder &b, Location l, ValueRange args) {
                linalg::YieldOp::create(b, l, ValueRange{arith::AddFOp::create(b, l, args[0], z)});
            });
        };
        // expects tenor x and contant scalar y
        auto add = [&](Value x, Value y) {
            return broadcastOne(x, [&](OpBuilder &b, Location l, ValueRange args) {
                linalg::YieldOp::create(b, l, ValueRange{arith::AddFOp::create(b, l, args[0], y)});
            });
        };
        // Compute the rational (sinh/cosh) polynomal approximation.
        // Coefficients computed through Remez Algorithm for sinh and
        // cosh then combined to calculate tanh.  (n)umerator (sinh)
        // coefficients:
        auto n0 = fpConst(-9.49066162e-1f);
        auto n1 = fpConst(-2.68447266e+1f);
        auto n2 = fpConst(-2.01115608e-2f);
        // (d)enominator (cosh) coefficients.
        auto d0 = fpConst(3.49853516e+1f);
        auto d1 = fpConst(8.07031250e+1f);
        // clamp range
        auto one = fpConst(1.0);
        auto mone = fpConst(-1.0);

        auto x = op.getInputs()[0];
        auto x2 = arith::MulFOp::create(rewriter, loc, x, x);
        auto sinh = fmaTensor(fmaConstant(n0, x2, n1), x2, n2);
        auto cosh = fmaTensor(add(x2, d0), x2, d1);
        auto recip = linalg::ReciprocalOp::create(
                         rewriter, loc, ValueRange{cosh}, ValueRange{emptyLike(cosh)}
        )
                         .getResult(0);
        auto quot = arith::MulFOp::create(rewriter, loc, sinh, recip);
        auto unclamped =
            arith::AddFOp::create(rewriter, loc, arith::MulFOp::create(rewriter, loc, quot, x), x);
        auto result =
            linalg::GenericOp::create(
                rewriter, loc, TypeRange{typeOf(x)}, ValueRange{unclamped},
                ValueRange{emptyLike(unclamped)},
                SmallVector<AffineMap>(2, AffineMap::getMultiDimIdentityMap(rank, ctx)),
                SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                [&](OpBuilder &b, Location l, ValueRange args) {
                    // // error: no member named 'ClampFOp' in namespace
                    // 'mlir::math' auto clamped = math::ClampFOp::create(b, l,
                    // res, mone, one);
                    auto tooLarge =
                        arith::CmpFOp::create(b, l, arith::CmpFPredicate::OLT, one, args[0]);
                    auto clampedAbove = arith::SelectOp::create(b, l, tooLarge, one, args[0]);
                    auto tooSmall =
                        arith::CmpFOp::create(b, l, arith::CmpFPredicate::OGT, mone, clampedAbove);
                    auto clamped = arith::SelectOp::create(b, l, tooSmall, mone, clampedAbove);
                    linalg::YieldOp::create(b, l, ValueRange{clamped});
                }
            ).getResult(0);

        rewriter.replaceOp(op, result);
        return success();
    }
};

// This doesn't get its own rewriter because it only works with
// negative inputs, which can't in general be inferred at compile
// time.  Use this function in other algos when you know what you are
// doing.
Value bfloatNegExp(Value x, PatternRewriter &rewriter, Location loc) {

    // useful meta variables
    auto ctx = rewriter.getContext();
    auto bf16TensorType = cast<RankedTensorType>(x.getType());
    auto bf16 = bf16TensorType.getElementType();
    auto shape = bf16TensorType.getShape();
    auto rank = bf16TensorType.getRank();
    auto i16 = IntegerType::get(ctx, 16);
    auto i16TensorType = RankedTensorType::get(shape, i16);
    auto i8 = IntegerType::get(ctx, 8);
    auto i1 = IntegerType::get(ctx, 1);
    auto indx = rewriter.getIndexType();

    // helper funcs to create constants
    auto tConst = [&](Type type, int constant) {
        return arith::ConstantOp::create(rewriter, loc, rewriter.getIntegerAttr(type, constant));
    };
    auto i16Tens = [&](std::vector<uint16_t> data) {
        return arith::ConstantOp::create(
                   rewriter, loc,
                   DenseIntElementsAttr::get(
                       RankedTensorType::get(data.size(), i16), ArrayRef<uint16_t>(data)
                   )
        )
            .getResult();
    };

    // helper functions to create linalg.generics
    auto emptyLike = [&](Value val) {
        return tensor::EmptyOp::create(rewriter, loc, val.getType(), ValueRange{});
    };
    auto broadcast = [&](Value v, Type t, auto func) {
        return linalg::GenericOp::create(
                   rewriter, loc, TypeRange{RankedTensorType::get(shape, t)}, ValueRange{v},
                   ValueRange{tensor::EmptyOp::create(rewriter, loc, shape, t)},
                   SmallVector<AffineMap>(2, AffineMap::getMultiDimIdentityMap(rank, ctx)),
                   SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                   [&](OpBuilder &b, Location l, ValueRange args) {
                       linalg::YieldOp::create(b, l, ValueRange{func(b, l, args)});
                   }
        ).getResult(0);
    };

    // create actual constants
    SmallVector<Value> sgnfBitmask;
    SmallVector<int> sgnfBitmaskData = {0b10000000, 0b1000000, 0b100000, 0b10000,
                                        0b1000,     0b100,     0b10,     0b1};
    SmallVector<Value> expoOffset;
    SmallVector<int> expoOffsetData = {7, 6, 5, 4, 3, 2, 1, 0};
    for (int i = 0; i < 8; i++) {
        sgnfBitmask.push_back(tConst(i8, sgnfBitmaskData[i]));
        expoOffset.push_back(tConst(i16, expoOffsetData[i]));
    }
    auto lutBits = i16Tens(
        {0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80,
         0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80,
         0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80,
         0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80,
         0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80,
         0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80,
         0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80,
         0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80,
         0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80,
         0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80,
         0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80,
         0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F7F, 0x3F7E, 0x3F7C, 0x3F78, 0x3F70, 0x3F62,
         0x3F47, 0x3F1B, 0x3EBC, 0x3E0B, 0x3C96, 0x39B0, 0x33F2, 0x2864, 0x114B, 0x0000, 0x0000,
         0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
         0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
         0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
         0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
         0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
         0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
         0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
         0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
         0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
         0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
         0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}
    );

    // allocate output tensors
    auto intX = tensor::BitcastOp::create(rewriter, loc, i16TensorType, x);

    auto dirtyExpo = broadcast(intX, i16, [&](OpBuilder &b, Location l, ValueRange args) {
        return arith::ShRUIOp::create(b, l, args[0], tConst(i16, 7));
    });
    // get rid of sign bit
    auto expo = broadcast(dirtyExpo, i16, [&](OpBuilder &b, Location l, ValueRange args) {
        return arith::AndIOp::create(b, l, args[0], tConst(i16, 0b0000000011111111));
    });
    auto mant = broadcast(intX, i8, [&](OpBuilder &b, Location l, ValueRange args) {
        return arith::TruncIOp::create(b, l, i8, args[0]);
    });
    auto sgnf = broadcast(mant, i8, [&](OpBuilder &b, Location l, ValueRange args) {
        return arith::OrIOp::create(b, l, args[0], tConst(i8, 0b10000000));
    });
    auto nanBoolMask = broadcast(intX, i1, [&](OpBuilder &b, Location l, ValueRange args) {
        // Sign extending comming in clutch.
        return arith::CmpIOp::create(
            b, l, arith::CmpIPredicate::ugt, args[0], tConst(i16, 0b1111111110000000)
        );
    });
    auto nanMask = broadcast(nanBoolMask, i16, [&](OpBuilder &b, Location l, ValueRange args) {
        // Sign extending comming in clutch.
        return arith::ExtSIOp::create(b, l, i16, args[0]);
    });
    auto outAlloc = broadcast(nanMask, i16, [&](OpBuilder &b, Location l, ValueRange args) {
        return arith::OrIOp::create(b, l, args[0], tConst(i16, 0b0011111110000000));
    });
    auto bfAlloc = tensor::BitcastOp::create(rewriter, loc, bf16TensorType, outAlloc);

    // calculate e^x from mantissa, exponent.  To do so, create 8
    // value per input value and reduce mul over those values (M ->
    // Mx8 -> M).  Ideally, we would do this in a loop to avoid
    // creating any Mx8 tensors, but our lowering isn't yet smart
    // enough to deal with that, so now we make some gigantic tensors.
    SmallVector<AffineExpr> dims;
    SmallVector<AffineExpr> expandedDims;
    SmallVector<int64_t> bigShape;
    AffineExpr d;
    for (int i = 0; i < rank; i++) {
        d = getAffineDimExpr(i, ctx);
        dims.push_back(d);
        expandedDims.push_back(d);
        bigShape.push_back(shape[i]);
    }
    d = getAffineDimExpr(rank, ctx);
    SmallVector<AffineExpr> trailDim = {d};
    expandedDims.push_back(d);
    bigShape.push_back(8);
    auto expandBroadcast = [&](Value mTensor, SmallVector<Value> eightTensor, Type t, auto func) {
        Value fullOutTens = tensor::EmptyOp::create(rewriter, loc, bigShape, t);
        auto partialOutTens = tensor::EmptyOp::create(rewriter, loc, shape, t);
        auto tType = RankedTensorType::get(shape, t);
        SmallVector<OpFoldResult> offsets, sizes, strides(rank + 1, rewriter.getIndexAttr(1));
        for (int64_t dim : shape)
            sizes.push_back(rewriter.getIndexAttr(dim));
        sizes.push_back(rewriter.getIndexAttr(1));
        for (int i = 0; i < 8; i++) {
            auto partialResult =
                linalg::GenericOp::create(
                    rewriter, loc, TypeRange{tType}, ValueRange{mTensor},
                    ValueRange{partialOutTens},
                    SmallVector<AffineMap>(2, AffineMap::getMultiDimIdentityMap(rank, ctx)),
                    SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                    [&](OpBuilder &b, Location l, ValueRange args) {
                        linalg::YieldOp::create(b, l, ValueRange{func(b, l, args, eightTensor[i])});
                    }
                ).getResult(0);
            offsets = SmallVector<OpFoldResult>(rank, rewriter.getIndexAttr(0));
            offsets.push_back(rewriter.getIndexAttr(i));
            fullOutTens = tensor::InsertSliceOp::create(
                rewriter, loc, partialResult, fullOutTens, offsets, sizes, strides
            );
        }
        return fullOutTens;
    };
    auto unmaskedLutIndex = expandBroadcast(
        expo, expoOffset, i16,
        [&](OpBuilder &b, Location l, ValueRange args, Value c) {
            return arith::AddIOp::create(b, l, args[0], c);
        }
    );
    auto indexRawMask = expandBroadcast(
        sgnf, sgnfBitmask, i8,
        [&](OpBuilder &b, Location l, ValueRange args, Value c) {
            return arith::AndIOp::create(b, l, args[0], c);
        }
    );
    auto bigI16TensorType = RankedTensorType::get(bigShape, i16);
    auto bigI8TensorType = RankedTensorType::get(bigShape, i8);
    auto staticBroadcast = [&](Value m8Tensor, auto func, auto type) {
        return linalg::GenericOp::create(
                   rewriter, loc, TypeRange{type}, ValueRange{m8Tensor},
                   ValueRange{tensor::EmptyOp::create(rewriter, loc, type, ValueRange{})},
                   SmallVector<AffineMap>(2, AffineMap::get(rank + 1, 0, expandedDims, ctx)),
                   SmallVector<utils::IteratorType>(rank + 1, utils::IteratorType::parallel),
                   [&](OpBuilder &b, Location l, ValueRange args) {
                       linalg::YieldOp::create(b, l, ValueRange{func(b, l, args)});
                   }
        ).getResult(0);
    };
    auto zero = tConst(i8, 0);
    auto one = tConst(i8, 1);
    auto index8Mask = staticBroadcast(
        indexRawMask,
        [&](OpBuilder &b, Location l, ValueRange args) {
            auto tooLarge = arith::CmpIOp::create(b, l, arith::CmpIPredicate::ult, one, args[0]);
            auto clampedAbove = arith::SelectOp::create(b, l, tooLarge, one, args[0]);
            auto tooSmall =
                arith::CmpIOp::create(b, l, arith::CmpIPredicate::ugt, zero, clampedAbove);
            return arith::SelectOp::create(b, l, tooSmall, zero, clampedAbove);
        },
        bigI8TensorType
    );
    auto indexMask = staticBroadcast(
        index8Mask,
        [&](OpBuilder &b, Location l, ValueRange args) {
            return arith::ExtSIOp::create(b, l, i16, args[0]);
        },
        bigI16TensorType
    );
    auto lutIndex = linalg::GenericOp::create(
                        rewriter, loc, TypeRange{bigI16TensorType},
                        ValueRange{indexMask, unmaskedLutIndex}, ValueRange{emptyLike(indexMask)},
                        SmallVector<AffineMap>(3, AffineMap::get(rank + 1, 0, expandedDims, ctx)),
                        SmallVector<utils::IteratorType>(rank + 1, utils::IteratorType::parallel),
                        [&](OpBuilder &b, Location l, ValueRange args) {
                            linalg::YieldOp::create(
                                b, l, ValueRange{arith::MulIOp::create(b, l, args[0], args[1])}
                            );
                        }
    ).getResult(0);

    auto lutVal = staticBroadcast(
        lutIndex,
        [&](OpBuilder &b, Location l, ValueRange args) {
            return tensor::ExtractOp::create(
                b, l, lutBits, ValueRange{arith::IndexCastOp::create(b, l, indx, args[0])}
            );
        },
        bigI16TensorType
    );
    auto bfVal =
        tensor::BitcastOp::create(rewriter, loc, RankedTensorType::get(bigShape, bf16), lutVal);
    // final reduce mul
    return linalg::ReduceOp::create(
               rewriter, loc, ValueRange{bfVal}, ValueRange{bfAlloc}, rank,
               [&](OpBuilder &b, Location l, ValueRange args) {
                   linalg::YieldOp::create(
                       b, l, ValueRange{arith::MulFOp::create(b, l, args[0], args[1])}
                   );
               }
    ).getResult(0);
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
        // auto i1 = rewriter.getIntegerType(1);
        // auto i1TensorType = RankedTensorType::get(bfTensorType.getShape(), i1);
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
        // // Any value (unsigned) greater than +inf is either a negative
        // // number, or nan. In either case, rsqrt of this is NaN.
        // const int16_t bf16Inf = 0x7f80;
        // auto inf = arith::ConstantOp::create(rewriter, loc, rewriter.getIntegerAttr(i16,
        // bf16Inf));
        // // rsqrt(neg) -> nan
        // auto nanBoolMask =
        //     broadcast(i, i1TensorType, [&](OpBuilder &b, Location l, ValueRange args) {
        //         return arith::CmpIOp::create(b, l, arith::CmpIPredicate::ugt, args[0], inf);
        //     });
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
        // auto nanMask =
        //     broadcast(nanBoolMask, i16TensorType, [&](OpBuilder &b, Location l, ValueRange args)
        //     {
        //         // Sign extending comming in clutch: (true) 0b1 -> 0b1111111111111111 (nan)
        //         return arith::ExtSIOp::create(b, l, i16, args[0]);
        //     });
        // auto i3 = arith::OrIOp::create(rewriter, loc, nanMask, i2);
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

void populateDecomposeLinalgOpsPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.insert<TensorBitcastPattern>(context);
    patterns.insert<BfloatGenericErfPattern>(context);
    patterns.insert<BfloatErfPattern>(context);
    patterns.insert<BfloatGenericTanhPattern>(context);
    patterns.insert<BfloatTanhPattern>(context);
    patterns.insert<BfloatDivfPattern>(context);
    patterns.insert<BfloatReciprocalPattern>(context);
    patterns.insert<QuantizedBatchMatmulPattern>(context);
    patterns.insert<PowToMulPattern>(context);
    patterns.insert<BfloatRsqrtPattern>(context);
    patterns.insert<CanonicalizeBF16CastComparePattern>(context);
}

} // namespace mlir::syna::torq
