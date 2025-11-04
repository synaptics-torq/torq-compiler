// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "OpPatternOptions.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "torq/Conversions/LinalgToTorqHL/MatchingFunctions.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

#define DEBUG_TYPE "linalg-torq-pre-pattern"

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

        auto emptyOp = rewriter.create<tensor::EmptyOp>(
            bitcastOp.getLoc(), resultType.getShape(), resultType.getElementType()
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
                auto castOp = nestedBuilder.create<arith::BitcastOp>(
                    loc, resultType.getElementType(), args[0]
                );
                nestedBuilder.create<linalg::YieldOp>(loc, ValueRange{castOp});
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

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {
        if (!cast<RankedTensorType>(srcOp.getType(0)).getElementType().isBF16() ||
            !isa_and_nonnull<arith::DivFOp>(getElementwiseBinaryOp(srcOp))) {
            return rewriter.notifyMatchFailure(srcOp, "Expected bf16 divf");
        }
        auto numerator = srcOp.getOperand(0);
        auto denominator = srcOp.getOperand(1);
        rewriter.replaceOp(
            srcOp,
            rewriter.create<arith::MulFOp>(
                srcOp.getLoc(), numerator,
                rewriter
                    .create<linalg::ReciprocalOp>(srcOp.getLoc(), denominator, srcOp.getOutputs())
                    .getResult(0)
            )
        );
        return success();
    }
};

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
            return rewriter
                .create<linalg::GenericOp>(
                    loc, TypeRange{tType(t)}, ValueRange{v},
                    ValueRange{rewriter.create<tensor::EmptyOp>(loc, shape, t)},
                    SmallVector<AffineMap>(2, AffineMap::getMultiDimIdentityMap(rank, ctx)),
                    SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                    [&](OpBuilder &b, Location l, ValueRange args) {
                        b.create<linalg::YieldOp>(l, ValueRange{func(b, l, args)});
                    }
                )
                .getResult(0);
        };
        auto i16Const = [&](int constant) {
            return rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(i16, constant));
        };
        auto andi = [&](int constant, Value val) {
            auto constVal = i16Const(constant);
            return broadcast(val, i16, [&](OpBuilder &b, Location l, ValueRange args) {
                return b.create<arith::AndIOp>(loc, constVal, args[0]);
            });
        };
        auto mul = [&](int constant, Value val) {
            auto constVal = i16Const(constant);
            return broadcast(val, i16, [&](OpBuilder &b, Location l, ValueRange args) {
                return b.create<arith::MulIOp>(loc, constVal, args[0]);
            });
        };
        auto sub = [&](int constant, Value val) {
            auto constVal = i16Const(constant);
            return broadcast(val, i16, [&](OpBuilder &b, Location l, ValueRange args) {
                return b.create<arith::SubIOp>(loc, constVal, args[0]);
            });
        };
        auto cmp = [&](int constant, Value val, arith::CmpIPredicate pred) {
            auto constVal = i16Const(constant);
            auto boolVal = broadcast(val, i1, [&](OpBuilder &b, Location l, ValueRange args) {
                return b.create<arith::CmpIOp>(l, pred, args[0], constVal);
            });
            return broadcast(boolVal, i16, [&](OpBuilder &b, Location l, ValueRange args) {
                return b.create<arith::ExtUIOp>(l, i16, args[0]);
            });
        };

        // bitcast to i16
        auto rawX = rewriter.create<tensor::BitcastOp>(loc, tType(i16), op.getInputs()[0]);

        // record sign bit
        auto xSign = andi(0b1000000000000000, rawX);

        // remove sign bit for following logic
        auto x = andi(0b0111111111111111, rawX);

        // create `or`-able mask for nan values (is this necessary?
        // My ML models never have nans to propagate...)
        auto inf = i16Const(0b0111111110000000);
        auto boolNanMask = broadcast(x, i1, [&](OpBuilder &b, Location l, ValueRange args) {
            return b.create<arith::CmpIOp>(l, arith::CmpIPredicate::ugt, args[0], inf);
        });
        auto nanMask = rewriter.create<arith::ExtSIOp>(loc, tType(i16), boolNanMask);

        // subnormal numbers are exactly the ones that map to
        // infinity.  Check which ones should map there.
        auto isSubnormal = cmp(0b0000000010000000, x, arith::CmpIPredicate::ult);

        // "big" means numbers that map to zero (we flush subnormals to zero).
        auto isNotBig = cmp(0b0111111010000000, x, arith::CmpIPredicate::ule);

        // extract exponent bits
        auto xExpo = andi(0b0111111110000000, x);

        // compute exponent of reciprocal from extracted reciprocal
        auto computedExpo = sub(0b0111111010000000, xExpo);

        // extract mantissa bits
        auto xMant = andi(0b0000000001111111, x);

        // we need to adjust our computed exponent for the special
        // case where mantissa == 0.
        auto specialMantissaValue = cmp(0b0000000000000000, xMant, arith::CmpIPredicate::eq);

        // where mantissa == 0, we need to add one to the exponent.
        auto specialExpoOffset = mul(0b0000000010000000, specialMantissaValue);
        auto realComputedExpo =
            rewriter.create<arith::AddIOp>(loc, specialExpoOffset, computedExpo);

        // Our magic LUT values!  See
        // scripts/torch/bfloat16_softmax.py to reproduce.
        std::vector<int8_t> lutData{
            // pad zeros for easier lowering
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
            // actual values
            0x00, 0x7E, 0x7C, 0x7A, 0x78, 0x76, 0x75, 0x73, 0x71, 0x6F, 0x6D, 0x6C, 0x6A, 0x68,
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

        // create LUT
        auto lut = rewriter.create<arith::ConstantOp>(
            loc, DenseIntElementsAttr::get(
                     RankedTensorType::get(256, rewriter.getI8Type()), ArrayRef<int8_t>(lutData)
                 )
        );

        // use mantissa as index to LUT
        auto xMant8 = rewriter.create<arith::TruncIOp>(loc, tType(i8), xMant);
        auto outputTens = rewriter.create<tensor::EmptyOp>(loc, tType(i8), ValueRange{});
        auto indexOffset = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(128));
        auto lutVal =
            rewriter
                .create<linalg::GenericOp>(
                    loc, TypeRange{tType(i8)}, ValueRange{xMant8}, ValueRange{outputTens},
                    SmallVector<AffineMap>(2, AffineMap::getMultiDimIdentityMap(rank, ctx)),
                    SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                    [&](OpBuilder &b, Location loc, ValueRange args) {
                        auto baseIndex =
                            b.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), args[0]);
                        auto index = b.create<arith::AddIOp>(loc, baseIndex, indexOffset);
                        auto extracted = b.create<tensor::ExtractOp>(loc, lut, ValueRange{index});
                        b.create<linalg::YieldOp>(loc, ValueRange{extracted});
                    }
                )
                .getResult(0);

        auto lutVal16 = rewriter.create<arith::ExtSIOp>(loc, tType(i16), lutVal);

        // combine computed exponent and mantissa
        auto computed = rewriter.create<arith::OrIOp>(loc, lutVal16, realComputedExpo);

        // There are a few possible inputs (big inputs) that must also
        // be sqashed to zero.
        auto computed2 = rewriter.create<arith::MulIOp>(loc, computed, isNotBig);

        // conversely, we need a way to ensure all subnormals map to
        // infinity.
        auto maybeInf = mul(0b0111111110000000, isSubnormal);

        // We want to combine our computed maybeInf and our current
        // computed using bfloat addition.  Bitcast it real quick,
        // add, and bitcast back.
        auto computedBfloat = rewriter.create<tensor::BitcastOp>(loc, bfTensorType, computed2);
        auto maybeInfBfloat = rewriter.create<tensor::BitcastOp>(loc, bfTensorType, maybeInf);
        auto realComputedBfloat =
            rewriter.create<arith::AddFOp>(loc, computedBfloat, maybeInfBfloat);
        auto realComputed = rewriter.create<tensor::BitcastOp>(loc, tType(i16), realComputedBfloat);

        // add back our sign bit we saved earlier
        auto combined = rewriter.create<arith::OrIOp>(loc, xSign, realComputed);
        // final nan propagation
        auto combined2 = rewriter.create<arith::OrIOp>(loc, combined, nanMask);

        // bitcast final value back to bfloat16
        auto bfBitcast = rewriter.create<tensor::BitcastOp>(loc, bfTensorType, combined2);

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
        rewriter.replaceOp(srcOp, rewriter.create<math::ErfOp>(srcOp.getLoc(), srcOp.getInputs()));
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

        // useful meta-values.
        auto loc = op.getLoc();
        auto ctx = rewriter.getContext();
        auto bfTensorType = cast<RankedTensorType>(op.getType());
        auto bf16 = bfTensorType.getElementType();
        auto rank = (size_t)bfTensorType.getRank();

        // helper functions.
        auto emptyLike = [&](Value v) {
            return rewriter.create<tensor::EmptyOp>(loc, v.getType(), ValueRange{});
        };
        auto broadcast = [&](Value v, auto func) {
            return rewriter
                .create<linalg::GenericOp>(
                    loc, TypeRange{v.getType()}, ValueRange{v}, ValueRange{emptyLike(v)},
                    SmallVector<AffineMap>(2, AffineMap::getMultiDimIdentityMap(rank, ctx)),
                    SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel), func
                )
                .getResult(0);
        };
        // only works if x and y are tensor and z is a scalar constants.
        auto fma = [&](Value x, Value y, Value z) {
            auto xy = rewriter.create<arith::MulFOp>(loc, x, y);
            return broadcast(xy, [&](OpBuilder &b, Location l, ValueRange args) {
                b.create<linalg::YieldOp>(l, ValueRange{b.create<arith::AddFOp>(l, args[0], z)});
            });
        };
        auto fmaConstant = [&](Value x, Value y, Value z) {
            auto xy = broadcast(y, [&](OpBuilder &b, Location l, ValueRange args) {
                b.create<linalg::YieldOp>(l, ValueRange{b.create<arith::MulFOp>(l, x, args[0])});
            });
            return broadcast(xy, [&](OpBuilder &b, Location l, ValueRange args) {
                b.create<linalg::YieldOp>(l, ValueRange{b.create<arith::AddFOp>(l, args[0], z)});
            });
        };
        // expects tenor x and contant scalar y
        auto mul = [&](Value x, Value y) {
            return broadcast(x, [&](OpBuilder &b, Location l, ValueRange args) {
                b.create<linalg::YieldOp>(l, ValueRange{b.create<arith::MulFOp>(l, args[0], y)});
            });
        };
        auto fpConst = [&](float fp) {
            return rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(bf16, fp));
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
        auto x2 = rewriter.create<arith::MulFOp>(loc, x, x);
        auto terms =
            fma(fma(fma(fma(fma(fmaConstant(c6, x2, c5), x2, c4), x2, c3), x2, c2), x2, c1), x2,
                c0);
        auto unclamped = rewriter.create<arith::MulFOp>(loc, terms, mul(x, scale));
        auto outAlloc = rewriter.create<tensor::EmptyOp>(loc, x.getType(), ValueRange{});
        auto erf =
            rewriter
                .create<linalg::GenericOp>(
                    loc, TypeRange{outAlloc.getType()}, ValueRange{unclamped}, ValueRange{outAlloc},
                    SmallVector<AffineMap>(2, AffineMap::getMultiDimIdentityMap(rank, ctx)),
                    SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                    [&](OpBuilder &b, Location loc, ValueRange args) {
                        auto tooLarge =
                            b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, one, args[0]);
                        auto clamped1 = b.create<arith::SelectOp>(loc, tooLarge, one, args[0]);
                        auto tooSmall =
                            b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, mone, clamped1);
                        auto clamped2 = b.create<arith::SelectOp>(loc, tooSmall, mone, clamped1);
                        b.create<linalg::YieldOp>(loc, ValueRange{clamped2});
                    }
                )
                .getResult(0);

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
            rewriter.create<linalg::TanhOp>(srcOp.getLoc(), srcOp.getInputs(), srcOp.getOutputs())
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

        auto ctx = rewriter.getContext();
        auto loc = op.getLoc();
        auto bf16TensType = cast<RankedTensorType>(op.getType(0));
        auto bf16 = bf16TensType.getElementType();
        auto rank = bf16TensType.getRank();

        auto emptyLike = [&](Value val) {
            return rewriter.create<tensor::EmptyOp>(loc, val.getType(), ValueRange{});
        };
        auto typeOf = [&](Value v) { return v.getType(); };
        auto fpConst = [&](float fp) {
            return rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(bf16, fp));
        };
        // broadcast over one tensor
        auto broadcastOne = [&](Value v, auto func) {
            return rewriter
                .create<linalg::GenericOp>(
                    loc, TypeRange{typeOf(v)}, ValueRange{v}, ValueRange{emptyLike(v)},
                    SmallVector<AffineMap>(2, AffineMap::getMultiDimIdentityMap(rank, ctx)),
                    SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel), func
                )
                .getResult(0);
        };
        // only works if x and y are tensor and z is a scalar constants.
        auto fmaTensor = [&](Value x, Value y, Value z) {
            auto xy = rewriter.create<arith::MulFOp>(loc, x, y);
            return broadcastOne(xy, [&](OpBuilder &b, Location l, ValueRange args) {
                b.create<linalg::YieldOp>(l, ValueRange{b.create<arith::AddFOp>(l, args[0], z)});
            });
        };
        auto fmaConstant = [&](Value x, Value y, Value z) {
            auto xy = broadcastOne(y, [&](OpBuilder &b, Location l, ValueRange args) {
                b.create<linalg::YieldOp>(l, ValueRange{b.create<arith::MulFOp>(l, x, args[0])});
            });
            return broadcastOne(xy, [&](OpBuilder &b, Location l, ValueRange args) {
                b.create<linalg::YieldOp>(l, ValueRange{b.create<arith::AddFOp>(l, args[0], z)});
            });
        };
        // expects tenor x and contant scalar y
        auto add = [&](Value x, Value y) {
            return broadcastOne(x, [&](OpBuilder &b, Location l, ValueRange args) {
                b.create<linalg::YieldOp>(l, ValueRange{b.create<arith::AddFOp>(l, args[0], y)});
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
        auto x2 = rewriter.create<arith::MulFOp>(loc, x, x);
        auto sinh = fmaTensor(fmaConstant(n0, x2, n1), x2, n2);
        auto cosh = fmaTensor(add(x2, d0), x2, d1);
        auto recip =
            rewriter
                .create<linalg::ReciprocalOp>(loc, ValueRange{cosh}, ValueRange{emptyLike(cosh)})
                .getResult(0);
        auto quot = rewriter.create<arith::MulFOp>(loc, sinh, recip);
        auto unclamped =
            rewriter.create<arith::AddFOp>(loc, rewriter.create<arith::MulFOp>(loc, quot, x), x);
        auto result =
            rewriter
                .create<linalg::GenericOp>(
                    loc, TypeRange{typeOf(x)}, ValueRange{unclamped},
                    ValueRange{emptyLike(unclamped)},
                    SmallVector<AffineMap>(2, AffineMap::getMultiDimIdentityMap(rank, ctx)),
                    SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                    [&](OpBuilder &b, Location l, ValueRange args) {
                        // // error: no member named 'ClampFOp' in namespace
                        // 'mlir::math' auto clamped = b.create<math::ClampFOp>(l,
                        // res, mone, one);
                        auto tooLarge =
                            b.create<arith::CmpFOp>(l, arith::CmpFPredicate::OLT, one, args[0]);
                        auto clampedAbove = b.create<arith::SelectOp>(l, tooLarge, one, args[0]);
                        auto tooSmall = b.create<arith::CmpFOp>(
                            l, arith::CmpFPredicate::OGT, mone, clampedAbove
                        );
                        auto clamped = b.create<arith::SelectOp>(l, tooSmall, mone, clampedAbove);
                        b.create<linalg::YieldOp>(l, ValueRange{clamped});
                    }
                )
                .getResult(0);

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
        return rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(type, constant));
    };
    auto i16Tens = [&](std::vector<uint16_t> data) {
        return rewriter
            .create<arith::ConstantOp>(
                loc, DenseIntElementsAttr::get(
                         RankedTensorType::get(data.size(), i16), ArrayRef<uint16_t>(data)
                     )
            )
            .getResult();
    };

    // helper functions to create linalg.generics
    auto emptyLike = [&](Value val) {
        return rewriter.create<tensor::EmptyOp>(loc, val.getType(), ValueRange{});
    };
    auto broadcast = [&](Value v, Type t, auto func) {
        return rewriter
            .create<linalg::GenericOp>(
                loc, TypeRange{RankedTensorType::get(shape, t)}, ValueRange{v},
                ValueRange{rewriter.create<tensor::EmptyOp>(loc, shape, t)},
                SmallVector<AffineMap>(2, AffineMap::getMultiDimIdentityMap(rank, ctx)),
                SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                [&](OpBuilder &b, Location l, ValueRange args) {
                    b.create<linalg::YieldOp>(l, ValueRange{func(b, l, args)});
                }
            )
            .getResult(0);
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
    auto intX = rewriter.create<tensor::BitcastOp>(loc, i16TensorType, x);

    auto dirtyExpo = broadcast(intX, i16, [&](OpBuilder &b, Location l, ValueRange args) {
        return b.create<arith::ShRUIOp>(l, args[0], tConst(i16, 7));
    });
    // get rid of sign bit
    auto expo = broadcast(dirtyExpo, i16, [&](OpBuilder &b, Location l, ValueRange args) {
        return b.create<arith::AndIOp>(l, args[0], tConst(i16, 0b0000000011111111));
    });
    auto mant = broadcast(intX, i8, [&](OpBuilder &b, Location l, ValueRange args) {
        return b.create<arith::TruncIOp>(l, i8, args[0]);
    });
    auto sgnf = broadcast(mant, i8, [&](OpBuilder &b, Location l, ValueRange args) {
        return b.create<arith::OrIOp>(l, args[0], tConst(i8, 0b10000000));
    });
    auto nanBoolMask = broadcast(intX, i1, [&](OpBuilder &b, Location l, ValueRange args) {
        // Sign extending comming in clutch.
        return b.create<arith::CmpIOp>(
            l, arith::CmpIPredicate::ugt, args[0], tConst(i16, 0b1111111110000000)
        );
    });
    auto nanMask = broadcast(nanBoolMask, i16, [&](OpBuilder &b, Location l, ValueRange args) {
        // Sign extending comming in clutch.
        return b.create<arith::ExtSIOp>(l, i16, args[0]);
    });
    auto outAlloc = broadcast(nanMask, i16, [&](OpBuilder &b, Location l, ValueRange args) {
        return b.create<arith::OrIOp>(l, args[0], tConst(i16, 0b0011111110000000));
    });
    auto bfAlloc = rewriter.create<tensor::BitcastOp>(loc, bf16TensorType, outAlloc);

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
        Value fullOutTens = rewriter.create<tensor::EmptyOp>(loc, bigShape, t);
        auto partialOutTens = rewriter.create<tensor::EmptyOp>(loc, shape, t);
        auto tType = RankedTensorType::get(shape, t);
        SmallVector<OpFoldResult> offsets, sizes, strides(rank + 1, rewriter.getIndexAttr(1));
        for (int64_t dim : shape)
            sizes.push_back(rewriter.getIndexAttr(dim));
        sizes.push_back(rewriter.getIndexAttr(1));
        for (int i = 0; i < 8; i++) {
            auto partialResult =
                rewriter
                    .create<linalg::GenericOp>(
                        loc, TypeRange{tType}, ValueRange{mTensor}, ValueRange{partialOutTens},
                        SmallVector<AffineMap>(2, AffineMap::getMultiDimIdentityMap(rank, ctx)),
                        SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                        [&](OpBuilder &b, Location l, ValueRange args) {
                            b.create<linalg::YieldOp>(
                                l, ValueRange{func(b, l, args, eightTensor[i])}
                            );
                        }
                    )
                    .getResult(0);
            offsets = SmallVector<OpFoldResult>(rank, rewriter.getIndexAttr(0));
            offsets.push_back(rewriter.getIndexAttr(i));
            fullOutTens = rewriter.create<tensor::InsertSliceOp>(
                loc, partialResult, fullOutTens, offsets, sizes, strides
            );
        }
        return fullOutTens;
    };
    auto unmaskedLutIndex = expandBroadcast(
        expo, expoOffset, i16,
        [&](OpBuilder &b, Location l, ValueRange args, Value c) {
            return b.create<arith::AddIOp>(l, args[0], c);
        }
    );
    auto indexRawMask = expandBroadcast(
        sgnf, sgnfBitmask, i8,
        [&](OpBuilder &b, Location l, ValueRange args, Value c) {
            return b.create<arith::AndIOp>(l, args[0], c);
        }
    );
    auto bigI16TensorType = RankedTensorType::get(bigShape, i16);
    auto bigI8TensorType = RankedTensorType::get(bigShape, i8);
    auto staticBroadcast = [&](Value m8Tensor, auto func, auto type) {
        return rewriter
            .create<linalg::GenericOp>(
                loc, TypeRange{type}, ValueRange{m8Tensor},
                ValueRange{rewriter.create<tensor::EmptyOp>(loc, type, ValueRange{})},
                SmallVector<AffineMap>(2, AffineMap::get(rank + 1, 0, expandedDims, ctx)),
                SmallVector<utils::IteratorType>(rank + 1, utils::IteratorType::parallel),
                [&](OpBuilder &b, Location l, ValueRange args) {
                    b.create<linalg::YieldOp>(l, ValueRange{func(b, l, args)});
                }
            )
            .getResult(0);
    };
    auto zero = tConst(i8, 0);
    auto one = tConst(i8, 1);
    auto index8Mask = staticBroadcast(
        indexRawMask,
        [&](OpBuilder &b, Location l, ValueRange args) {
            auto tooLarge = b.create<arith::CmpIOp>(l, arith::CmpIPredicate::ult, one, args[0]);
            auto clampedAbove = b.create<arith::SelectOp>(l, tooLarge, one, args[0]);
            auto tooSmall =
                b.create<arith::CmpIOp>(l, arith::CmpIPredicate::ugt, zero, clampedAbove);
            return b.create<arith::SelectOp>(l, tooSmall, zero, clampedAbove);
        },
        bigI8TensorType
    );
    auto indexMask = staticBroadcast(
        index8Mask,
        [&](OpBuilder &b, Location l, ValueRange args) {
            return b.create<arith::ExtSIOp>(l, i16, args[0]);
        },
        bigI16TensorType
    );
    auto lutIndex =
        rewriter
            .create<linalg::GenericOp>(
                loc, TypeRange{bigI16TensorType}, ValueRange{indexMask, unmaskedLutIndex},
                ValueRange{emptyLike(indexMask)},
                SmallVector<AffineMap>(3, AffineMap::get(rank + 1, 0, expandedDims, ctx)),
                SmallVector<utils::IteratorType>(rank + 1, utils::IteratorType::parallel),
                [&](OpBuilder &b, Location l, ValueRange args) {
                    b.create<linalg::YieldOp>(
                        l, ValueRange{b.create<arith::MulIOp>(l, args[0], args[1])}
                    );
                }
            )
            .getResult(0);

    auto lutVal = staticBroadcast(
        lutIndex,
        [&](OpBuilder &b, Location l, ValueRange args) {
            return b.create<tensor::ExtractOp>(
                l, lutBits, ValueRange{b.create<arith::IndexCastOp>(l, indx, args[0])}
            );
        },
        bigI16TensorType
    );
    auto bfVal =
        rewriter.create<tensor::BitcastOp>(loc, RankedTensorType::get(bigShape, bf16), lutVal);
    // final reduce mul
    return rewriter
        .create<linalg::ReduceOp>(
            loc, ValueRange{bfVal}, ValueRange{bfAlloc}, rank,
            [&](OpBuilder &b, Location l, ValueRange args) {
                b.create<linalg::YieldOp>(
                    l, ValueRange{b.create<arith::MulFOp>(l, args[0], args[1])}
                );
            }
        )
        .getResult(0);
};

struct BfloatSoftmaxPattern : OpRewritePattern<linalg::SoftmaxOp> {
    BfloatSoftmaxPattern(MLIRContext *context)
        : OpRewritePattern<linalg::SoftmaxOp>(context, /*benefit=*/0) {
        setDebugName("BfloatSoftmaxPattern");
    }
    LogicalResult matchAndRewrite(linalg::SoftmaxOp op, PatternRewriter &rewriter) const override {

        // checks
        if (!cast<RankedTensorType>(op.getType(0)).getElementType().isBF16()) {
            return rewriter.notifyMatchFailure(op, "Expected bf16 output");
        }

        // Matched!  Define some useful shorthands.
        auto loc = op.getLoc();
        auto ctx = rewriter.getContext();

        // get some covenient type constants.
        auto bf16TensType = cast<RankedTensorType>(op.getType(0));
        auto bf16 = bf16TensType.getElementType();
        auto shape = bf16TensType.getShape();
        auto rank = bf16TensType.getRank();
        auto i16 = IntegerType::get(ctx, 16);
        auto softmaxDim = op.getDimension();

        // Allocate a tensor like input.
        auto emptyLike = [&](Value val) {
            return rewriter.create<tensor::EmptyOp>(loc, val.getType(), ValueRange{});
        };

        // Generate affine maps and iterator lists for the various
        // operations to be performed.
        SmallVector<AffineExpr> dims;
        SmallVector<AffineExpr> nonSoftmaxDims;
        SmallVector<int64_t> reduceMaxShape;
        AffineExpr d;
        for (int i = 0; i < rank; i++) {
            d = getAffineDimExpr(i, ctx);
            dims.push_back(d);
            if (i == softmaxDim) {
            }
            else {
                nonSoftmaxDims.push_back(d);
                reduceMaxShape.push_back(shape[i]);
            }
        }
        SmallVector<AffineExpr> trailDim = {getAffineDimExpr(rank, ctx)};

        // It was difficult getting -inf in bf16.  This simplifies away during iree-opt.
        auto inf16 = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerAttr(i16, 0b0111111110000000)
        );
        auto inf = rewriter.create<arith::BitcastOp>(loc, bf16, inf16);
        auto bfZero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(bf16));
        auto negInf = rewriter.create<arith::SubFOp>(loc, bfZero, inf);

        // subtract off bias
        auto x = op.getInput();
        auto reduceMaxType = RankedTensorType::get(reduceMaxShape, bf16);
        auto maxAlloc =
            rewriter
                .create<linalg::FillOp>(
                    loc, ValueRange{negInf},
                    ValueRange{rewriter.create<tensor::EmptyOp>(loc, reduceMaxType, ValueRange{})}
                )
                .getResult(0);
        auto max = rewriter
                       .create<linalg::ReduceOp>(
                           loc, x, maxAlloc, softmaxDim,
                           [&](OpBuilder &b, Location l, ValueRange args) {
                               b.create<linalg::YieldOp>(
                                   l, ValueRange{b.create<arith::MaximumFOp>(l, args[0], args[1])}
                               );
                           }
                       )
                       .getResult(0);
        auto biasedX =
            rewriter
                .create<linalg::GenericOp>(
                    loc, TypeRange{bf16TensType}, ValueRange{max, x}, ValueRange{emptyLike(x)},
                    (SmallVector<AffineMap>){
                        AffineMap::get(rank, 0, nonSoftmaxDims, ctx),
                        AffineMap::get(rank, 0, dims, ctx), AffineMap::get(rank, 0, dims, ctx)
                    },
                    SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                    [&](OpBuilder &b, Location l, ValueRange args) {
                        auto biased = b.create<arith::SubFOp>(l, args[1], args[0]);
                        b.create<linalg::YieldOp>(l, ValueRange{biased});
                    }
                )
                .getResult(0);

        // perform e^x under the assumption that x <= 0 since we subtracted max
        auto ex = bfloatNegExp(biasedX, rewriter, loc);

        // obtain denominator for final softmax operation.
        auto reduceSumType = reduceMaxType;
        auto denomAlloc =
            rewriter
                .create<linalg::FillOp>(
                    loc,
                    ValueRange{rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(bf16))},
                    ValueRange{rewriter.create<tensor::EmptyOp>(loc, reduceSumType, ValueRange{})}
                )
                .getResult(0);
        auto denom = rewriter
                         .create<linalg::ReduceOp>(
                             loc, ex, denomAlloc, softmaxDim,
                             [&](OpBuilder &b, Location l, ValueRange args) {
                                 b.create<linalg::YieldOp>(
                                     l, ValueRange{b.create<arith::AddFOp>(l, args[0], args[1])}
                                 );
                             }
                         )
                         .getResult(0);
        auto recip =
            rewriter.create<linalg::ReciprocalOp>(loc, ValueRange{denom}, ValueRange{denom})
                .getResult(0);

        // final val.  Woot!
        auto softmax =
            rewriter
                .create<linalg::GenericOp>(
                    loc, TypeRange{bf16TensType}, ValueRange{recip, ex}, ValueRange{emptyLike(ex)},
                    (SmallVector<AffineMap>){
                        AffineMap::get(rank, 0, nonSoftmaxDims, ctx),
                        AffineMap::get(rank, 0, dims, ctx), AffineMap::get(rank, 0, dims, ctx)
                    },
                    SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                    [&](OpBuilder &b, Location l, ValueRange args) {
                        auto result = b.create<arith::MulFOp>(l, args[0], args[1]);
                        b.create<linalg::YieldOp>(l, ValueRange{result});
                    }
                )
                .getResult(0);

        rewriter.replaceOp(op, softmax);
        return success();
    }
};

Value applyPaddingIfNeeded(
    Value input, RankedTensorType inputType, PatternRewriter &rewriter, Location loc,
    const PaddingInfo &padInfo
) {
    int64_t padLeft = padInfo.lrtbPad[0];
    int64_t padRight = padInfo.lrtbPad[1];

    bool needsPad = (padLeft > 1 || padRight > 1);
    if (!needsPad)
        return input;

    auto inputShape = inputType.getShape();
    SmallVector<int64_t> paddedShape(inputShape.begin(), inputShape.end());
    // paddedShape.back() += padLeft + padRight;

    auto elemType = inputType.getElementType();

    if (!elemType.isInteger(32)) {
        llvm::errs() << "Unsupported element type for FillOp (only i32 supported)\n";
        return input;
    }

    auto paddedType = RankedTensorType::get(paddedShape, elemType);
    auto initTensor = rewriter.create<tensor::EmptyOp>(loc, paddedShape, elemType);

    auto fillOp = rewriter.create<torq_hl::FillOp>(
        loc,
        paddedType,             // Output type
        initTensor.getResult(), // Init tensor value
        rewriter.getI32IntegerAttr(padInfo.padValue)
    );
    // Create InsertSliceOp: insert input tensor into padded tensor at correct offset
    SmallVector<OpFoldResult> offsets(inputShape.size(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides(inputShape.size(), rewriter.getIndexAttr(1));

    for (int64_t dim : inputShape)
        sizes.push_back(rewriter.getIndexAttr(dim));

    offsets.back() = rewriter.getIndexAttr(padLeft); // Insert input starting at padLeft

    Value inserted = rewriter.create<tensor::InsertSliceOp>(
        loc, input, fillOp.getOutput(), offsets, sizes, strides
    );

    return inserted;
}

void dumpModuleToFile(Operation *op, StringRef filename) {
    // Traverse upward to find the parent module
    mlir::Operation *parent = op;
    while (parent && !llvm::isa<mlir::ModuleOp>(parent))
        parent = parent->getParentOp();

    if (!parent) {
        llvm::errs() << "Failed to find parent ModuleOp for dumping IR.\n";
        return;
    }

    std::error_code ec;
    llvm::raw_fd_ostream file(filename, ec, llvm::sys::fs::OF_None);
    if (ec) {
        llvm::errs() << "Failed to open file " << filename << ": " << ec.message() << "\n";
        return;
    }

    // Dump the whole module IR
    parent->print(file, mlir::OpPrintingFlags().useLocalScope());
}

template <class LinalgConvOp, class TorqConvOp>
struct Conv2dConvert : public OpRewritePattern<LinalgConvOp> {
  private:
    using MatchFn = bool(
        ArrayRef<int64_t> inputShape, ArrayRef<int64_t> weightShape, ArrayRef<int64_t> padShape
    );

    const int _channelDim;          // Channel dimension index in data shape
    const Permutation _dataPerm;    // Dim permutation for data transpose
    const Permutation _weightsPerm; // Weights permutation for weight transpose
    const int _shift8b;             // Scale shift for 8-bit integer operations
    const int _shift16b;            // Scale shift for 16-bit integer operations
    MatchFn *_matchFn;              // Function to match the convolution operation
    const bool _markFuseGroups;     // When true, mark the TI operations, don't convert.

  public:
    using OpRewritePattern<LinalgConvOp>::OpRewritePattern;
    Conv2dConvert(
        MLIRContext *context, int channelDim, const Permutation &dataPerm,
        const Permutation &weightsPerm, int shift8b, int shift16b, MatchFn *matchFn,
        bool markFuseGroups
    )
        : OpRewritePattern<LinalgConvOp>(context), _channelDim(channelDim), _dataPerm(dataPerm),
          _weightsPerm(weightsPerm), _shift8b(shift8b), _shift16b(shift16b), _matchFn(matchFn),
          _markFuseGroups(markFuseGroups) {}

    LogicalResult matchAndRewrite(LinalgConvOp convOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(convOp)) {
            return rewriter.notifyMatchFailure(convOp, "Already marked");
        }

        constexpr int groups = 1; // We don't use it
        const auto loc = convOp.getLoc();

        // Get the input, weights, and output of the original operation
        Value input = convOp.image();
        Value weights = convOp.filter();
        Value output = convOp.getResultTensors()[0];

        auto inputType = llvm::cast<RankedTensorType>(input.getType());
        auto shape = inputType.getShape();
        auto weightType = llvm::cast<RankedTensorType>(weights.getType());
        auto weightShape = weightType.getShape();

        bool isConv1D = (inputType.getRank() == 4 && shape[1] == 1);
        if (isConv1D) {
            return rewriteAsConv1D(convOp, rewriter);
        }

        // Fold padding if present
        PaddingInfo padInfo = foldBackwardPadding(input, rewriter);

        // Check if we can support this layer
        if (_matchFn && !_matchFn(shape, weightShape, padInfo.lrtbPad)) {
            return rewriter.notifyMatchFailure(
                convOp, "Conv does not match expected kernel dimension or padding"
            );
        }

        // Fold any per-channel bias
        const auto outType = cast<RankedTensorType>(output.getType());
        const int outChannelCount = outType.getShape()[_channelDim];
        bool isInt = outType.getElementType().isInteger();
        VectorIntOrFloat bias(outChannelCount, isInt);
        while (foldForwardPerChannelAdd(output, _channelDim, bias)) {
        }

        // Fold operations that take care of zero-point in weight quantization if present
        int weightZp = foldForwardWeightZp(output);

        // Fold any additional per-channel bias
        while (foldForwardPerChannelAdd(output, _channelDim, bias)) {
        }

        // Fold scale and clamp. This is mandatory for integer operations.
        ScaleClampInfo scInfo =
            foldForwardScaleClamp(output, outChannelCount, _shift8b, _shift16b, false);
        if (!scInfo && isInt) {
            return rewriter.notifyMatchFailure(
                convOp, "Expected scale and clamp info for integer operations"
            );
        }

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {input}, rewriter,
                convOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        // Convert weights to the required format
        DenseIntOrFPElementsAttr weightAttr;
        auto transposedWeights = transposeValue(weights, _weightsPerm, loc, rewriter);
        if (weightZp != 0) {
            // Divide weights and wZp by two so we can safely add them together without overlflow
            constexpr int scaleFactor = 2;
            transposedWeights =
                rescaleValue(transposedWeights, scaleFactor, -weightZp / 2, loc, rewriter);
            // Same for the bias
            for (auto &val : bias.ints) {
                val /= scaleFactor;
            }
            // Reduce the scaling shift to compensate
            // (We could multiply the scaling factor by 2 instead, but that could cause overflow)
            scInfo.scaleShift -= 1;
        }
        weightAttr = computeConstant(transposedWeights);
        auto torqWeights = createConst(weightAttr, rewriter, loc);
        if (!torqWeights) {
            return rewriter.notifyMatchFailure(
                convOp, "Failed to create constant for transposed weights"
            );
        }

        // Prepare bias (and scale for integer ops)
        auto biasScale = isInt ? createConst(interleave(bias.ints, scInfo.scaleNpu), rewriter, loc)
                               : createConst(bias.floats, rewriter, loc);

        // Generate torq_hl op with input/output in the expected format
        input = transposeValue(input, _dataPerm, loc, rewriter);
        auto torqOutType = transposeType(output.getType(), _dataPerm);
        bool nhwcInput = _channelDim == 3 && _dataPerm.empty();
        auto torqConvOp = rewriter.create<TorqConvOp>(
            loc, torqOutType, createInitTensor(convOp, rewriter, torqOutType), padInfo.padValue, 0,
            scInfo.zp, scInfo.min, scInfo.max, scInfo.scaleShift, groups, padInfo.lrtbPad,
            attrValues(convOp.getStrides()), attrValues(convOp.getDilations()),
            torq_hl::VectorizationModeEnum::None, torqWeights, biasScale, input, nhwcInput
        );
        auto torqOut = transposeValue(torqConvOp.getOutput(), _dataPerm.reverse(), loc, rewriter);
        rewriter.replaceOp(output.getDefiningOp(), torqOut);
        return success();
    }

  private:
    template <typename LinalgConv>
    LogicalResult rewriteAsConv1D(LinalgConv convOp, PatternRewriter &rewriter) const {
        if (!isa<linalg::Conv2DNhwcHwcfOp>(convOp)) {
            return rewriter.notifyMatchFailure(
                convOp, "Only linalg::Conv2DNhwcHwcfOp can be rewritten as Conv1D"
            );
        }
        if (_markFuseGroups && isMarkedFuseGroup(convOp)) {
            return rewriter.notifyMatchFailure(convOp, "Already marked");
        }

        auto loc = convOp.getLoc();
        constexpr int weightZp = 0;
        constexpr int groups = 1;
        Value input = convOp.image();
        Value weights = convOp.filter();
        Value output = convOp.getResultTensors()[0];
        ::mlir::DenseIntElementsAttr stridesAttr = convOp.getStrides();
        auto strideValue = stridesAttr.getValues<int64_t>()[1];

        auto inputType = cast<RankedTensorType>(input.getType());
        auto outputType = cast<RankedTensorType>(output.getType());
        auto outElemType = outputType.getElementType();
        bool isInt = outElemType.isInteger();
        int outChannels = outputType.getShape()[_channelDim];

        VectorIntOrFloat bias(outChannels, isInt);
        while (foldForwardPerChannelAdd(output, _channelDim, bias)) {
        }

        ScaleClampInfo scInfo = foldForwardScaleClamp(output, outChannels, _shift8b, _shift16b);
        if (!scInfo && isInt)
            return failure();

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {input}, rewriter,
                convOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        auto weightAttr = computeConstant(transposeValue(weights, _weightsPerm, loc, rewriter));
        auto torqWeights = createConst(weightAttr, rewriter, loc);
        // TODO: Torq weights should be reorderes in multiple channels cases;
        if (!torqWeights)
            return failure();

        auto biasScale = isInt ? createConst(interleave(bias.ints, scInfo.scaleNpu), rewriter, loc)
                               : createConst(bias.floats, rewriter, loc);

        // Note: op is Conv2DNhwcHwcfOp
        int64_t batch = inputType.getShape()[0];
        int64_t channels = inputType.getShape()[3];
        int64_t out_len = outputType.getShape()[2];
        int64_t outputChannels = outputType.getShape()[3];

        auto weightType = cast<RankedTensorType>(weights.getType());
        int64_t filter_len = weightType.getShape()[1];

        int64_t op_rows = filter_len;
        int64_t op_cols = out_len;

        llvm::SmallVector<int64_t> transposedShape = {batch, channels, op_rows, op_cols};
        RankedTensorType transposedType =
            RankedTensorType::get(transposedShape, inputType.getElementType());

        llvm::SmallVector<int64_t> permVals = {1, 0};
        auto permAttr = mlir::DenseI64ArrayAttr::get(rewriter.getContext(), permVals);

        auto torqOutType = transposeType(output.getType(), _dataPerm);

        // Decide whether to use Conv1D with reduction or TransposeReshape + Conv1D
        // The former is completely generic but probably less efficient for single-channel cases
        // The latter is more efficient but only works for single-channel input and outputs.
        bool useConv1dWithReduce = channels > 1 || outputChannels > 1;
        if (useConv1dWithReduce) {
            input = transposeValue(input, _dataPerm, loc, rewriter);
            // Create type for Conv1D output with an extra dimension at the end.
            // This will be reduced later with linalg.reduce.
            llvm::SmallVector<int64_t> torqOutShape(
                torqOutType.getShape().begin(), torqOutType.getShape().end()
            );
            torqOutShape.push_back(filter_len);
            torqOutType = RankedTensorType::get(torqOutShape, torqOutType.getElementType());
        }
        else {
            auto transposeReshape = rewriter.create<torq_hl::TransposeReshapeOp>(
                loc, transposedType, createInitTensor(convOp, rewriter, transposedType),
                attrValues(convOp.getStrides()), weightType.getShape(), permAttr, input
            );
            input = transposeReshape.getOutput();
            // Reset stride to 1 for Conv1DOp as the actual stride is handled in TransposeReshape
            strideValue = 1;
        }

        llvm::SmallVector<int64_t> zeroPad(4, 0);
        llvm::SmallVector<int64_t> stride = {strideValue};

        auto torqConv1Op = rewriter.create<torq_hl::Conv1DOp>(
            loc, torqOutType, createInitTensor(convOp, rewriter, torqOutType), 0, weightZp,
            scInfo.zp, scInfo.min, scInfo.max, scInfo.scaleShift, groups, zeroPad, stride,
            attrValues(convOp.getDilations()), torq_hl::VectorizationModeEnum::None, torqWeights,
            biasScale, input
        );
        Value torqOut = torqConv1Op.getOutput();

        if (useConv1dWithReduce) {
            // Add linalg.reduce to remove the extra dimension
            // Create reducedType from torqOutType by removing the last dimension
            auto reducedShape = torqOutType.getShape().drop_back();

            // Create a tensor filled with zeros of type torqOutType.getElementType()
            Value zeroValue = createZeroConstant(rewriter, loc, torqOutType.getElementType());
            auto cEmpty =
                rewriter.create<tensor::EmptyOp>(loc, reducedShape, torqOutType.getElementType());
            Value zeroTensor =
                rewriter.create<linalg::FillOp>(loc, ValueRange{zeroValue}, ValueRange{cEmpty})
                    .result();
            linalg::ReduceOp reduceOp = rewriter.create<linalg::ReduceOp>(
                loc, ValueRange{torqOut}, ValueRange{zeroTensor}, 4,
                [&](OpBuilder &b, Location l, ValueRange args) {
                    b.create<linalg::YieldOp>(
                        l, ValueRange{b.create<arith::AddFOp>(l, args[0], args[1])}
                    );
                }
            );

            torqOut = reduceOp->getResult(0);
        }
        // // Overwrite torqOut with init tensor for debugging
        // torqOut = createInitTensor(convOp, rewriter, cast<RankedTensorType>(torqOut.getType()));
        // // Fill input with 1s for debugging
        // torqOut = rewriter.create<torq_hl::FillOp>(
        //     loc, cast<RankedTensorType>(torqOut.getType()), torqOut,
        //     rewriter.getI32IntegerAttr(/*0x3f800000*//*0x00003f80*/0)
        // ).getOutput();

        torqOut = transposeValue(torqOut, _dataPerm.reverse(), loc, rewriter);

        rewriter.replaceOp(output.getDefiningOp(), torqOut);
        return success();
    }
};

struct PoolingNhwcMaxOpConversion : public OpRewritePattern<linalg::PoolingNhwcMaxOp> {
  private:
    const bool _markFuseGroups;

  public:
    using OpRewritePattern::OpRewritePattern;
    PoolingNhwcMaxOpConversion(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::PoolingNhwcMaxOp srcOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(srcOp)) {
            return rewriter.notifyMatchFailure(srcOp, "Already marked");
        }

        if (srcOp.getInputs().size() != 2 || srcOp.getResults().size() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "Expects 2 inputs and 1 output");
        }
        auto loc = srcOp.getLoc();
        Value input = srcOp.getInputs()[0];
        Value output = srcOp.getResults()[0];

        auto attrStrides = attrValues(srcOp.getStrides());
        if (attrStrides.size() != 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected exactly two strides for PoolingNhwcMaxOp"
            );
        }
        if (attrStrides[0] > 2 || attrStrides[1] > 2) {
            return rewriter.notifyMatchFailure(srcOp, "Expected strides <= 2 for PoolingNhwcMaxOp");
        }

        const std::vector<int32_t> bias = {0};
        const std::vector<int32_t> scale = {1};
        const std::vector<int8_t> weight = {1};

        PaddingInfo padInfo = foldBackwardPadding(input, rewriter);

        auto kernels = mlir::cast<RankedTensorType>(srcOp.getInputs()[1].getType()).getShape();
        if (kernels.size() != 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected exactly two kernel sizes for PoolingNhwcMaxOp"
            );
        }

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {input}, rewriter,
                srcOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        auto srcResultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

        auto dataPerm =
            srcResultType.getRank() == 4 ? Permutation::nhwc2nchw() : Permutation::none();

        input = transposeValue(input, dataPerm, loc, rewriter);
        srcResultType = transposeType(srcResultType, dataPerm);

        auto maxpoolOp = rewriter.create<torq_hl::MaxPool2dOp>(
            loc, srcResultType, createInitTensor(srcOp, rewriter, srcResultType), padInfo.padValue,
            attrStrides, padInfo.lrtbPad, kernels,
            createI8Const(rewriter, srcOp, weight, llvm::ArrayRef<int64_t>{1, 1, 1, 1}),
            createI32Const(rewriter, srcOp, interleave(bias, scale)), input
        );
        auto result = transposeValue(maxpoolOp.getOutput(), dataPerm.reverse(), loc, rewriter);

        rewriter.replaceOp(output.getDefiningOp(), result);

        return success();
    }
};

struct FCMatmulOpConversion : public OpRewritePattern<linalg::MatmulOp> {
  private:
    const bool _markFuseGroups;

  public:
    using OpRewritePattern::OpRewritePattern;
    FCMatmulOpConversion(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::MatmulOp srcOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(srcOp)) {
            return rewriter.notifyMatchFailure(srcOp, "Already marked");
        }

        const auto loc = srcOp.getLoc();
        if (srcOp.getInputs().size() != 2 || srcOp.getResults().size() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "Expects 2 input and 1 output");
        }

        Value inputA = srcOp.getInputs()[0];
        Value inputB = srcOp.getInputs()[1]; // weights
        Value output = srcOp.getResultTensors()[0];

        auto inputAType = llvm::cast<RankedTensorType>(inputA.getType());
        auto inputBType = llvm::cast<RankedTensorType>(inputB.getType());
        auto outputType = llvm::cast<RankedTensorType>(output.getType());
        if (inputAType.getRank() != 2 || inputBType.getRank() != 2 || outputType.getRank() != 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "FCMatmulOpConversion expects 2D inputs and outputs"
            );
        }
        auto inputAShape = inputAType.getShape();
        if (inputAShape[0] != 1) {
            return rewriter.notifyMatchFailure(
                srcOp, "FCMatmulOpConversion expects inputA shape[0] == 1"
            );
        }

        auto outputChannelCount = outputType.getShape()[1];
        bool isInt = outputType.getElementType().isInteger();
        VectorIntOrFloat bias(outputChannelCount, isInt);
        int32_t input_zp = 0;
        while (foldForwardPerChannelAdd(output, 1, bias, &input_zp)) {
        }

        // check if output user is expand_shape
        if (output.hasOneUse() && (isa<tensor::ExpandShapeOp>(*output.getUsers().begin()) ||
                                   isCollapseOrExpandShapeGeneric(*output.getUsers().begin()))) {
            auto op = *output.getUsers().begin();
            if (_markFuseGroups) {
                output = op->getResult(0);
            }
            else {
                op->getResult(0).replaceAllUsesWith(output);
                rewriter.eraseOp(op);
            }
        }

        ScaleClampInfo scInfo = foldForwardScaleClamp(output, outputChannelCount, 20, 12);
        if (!scInfo && isInt) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected scale and clamp info for integer operations"
            );
        }

        // check if output is a tensor::CollapseShapeOp
        if (output.hasOneUse() && (isa<tensor::CollapseShapeOp>(*output.getUsers().begin()) ||
                                   isCollapseOrExpandShapeGeneric(*output.getUsers().begin()))) {
            output = output.getUsers().begin()->getResult(0);
        }

        // Prepare weights
        if (auto transposeOp = dyn_cast<linalg::TransposeOp>(inputB.getDefiningOp())) {
            inputB = transposeOp.getInput();
            // NOTE: inputB changed, re-get its type if need to process related
        }

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {inputA, inputB}, rewriter,
                srcOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        auto weightAttr = computeConstant(inputB);
        auto torqWeights = createConst(weightAttr, rewriter, loc);
        if (!torqWeights) {
            return rewriter.notifyMatchFailure(
                srcOp, "Failed to create constant for transposed weights"
            );
        }
        // Prepare bias (and scale for integer ops)
        auto biasScale = isInt ? createConst(interleave(bias.ints, scInfo.scaleNpu), rewriter, loc)
                               : createConst(bias.floats, rewriter, loc);

        // get new output type as above various changes for output
        outputType = llvm::cast<RankedTensorType>(output.getType());

        auto fcOp = rewriter.create<torq_hl::FullyConnectedOp>(
            loc, outputType, createInitTensor(srcOp, rewriter, outputType), input_zp,
            0, // weight zp
            scInfo.zp, scInfo.min, scInfo.max, scInfo.scaleShift,
            torq_hl::VectorizationModeEnum::None, torqWeights, biasScale, inputA
        );
        rewriter.replaceOp(output.getDefiningOp(), fcOp.getOutput());

        return success();
    }
};

struct Conv2DMatmulOpConversion : public OpRewritePattern<linalg::MatmulOp> {
  private:
    const bool _markFuseGroups;

  public:
    using OpRewritePattern::OpRewritePattern;
    Conv2DMatmulOpConversion(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern(context), _markFuseGroups(markFuseGroups) {}

    Value convertWeights(
        mlir::linalg::MatmulOp srcOp, mlir::DenseIntOrFPElementsAttr weightAttr,
        PatternRewriter &rewriter
    ) const {
        // Reorder weights to OIHW
        auto weightElemType = weightAttr.getElementType();
        auto weightShape = dyn_cast<ShapedType>(weightAttr.getType()).getShape();

        // Validate expected shape: [OC, IC] for matmul-style
        assert(weightShape.size() == 2);

        // Assume shape was originally [OC, IC] from matmul-style
        int on = weightShape[0]; // OC
        int in = weightShape[1]; // IC
        int hn = 1;
        int wn = 1;
        std::vector<int64_t> weight_shape{on, in, hn, wn};

        if (weightElemType.isBF16()) {
            auto bfVals = weightAttr.getValues<APFloat>();
            const std::vector<APFloat> bfVec(bfVals.begin(), bfVals.end());
            std::vector<APFloat> reordered = get_weights_OIHW<APFloat>(bfVec, on, hn, wn, in);
            return createFConst(rewriter, srcOp, reordered, weight_shape);
        }
        else if (weightElemType.isInteger(8)) {
            auto rawVals = weightAttr.getValues<int8_t>();
            std::vector<int8_t> reordered(rawVals.begin(), rawVals.end());
            reordered = get_weights_OIHW<int8_t>(reordered, on, hn, wn, in);
            return createI8Const(rewriter, srcOp, reordered, weight_shape);
        }
        else {
            assert(false && "Unsupported weight type");
        }
    }

    LogicalResult
    matchAndRewrite(linalg::MatmulOp srcOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(srcOp)) {
            return rewriter.notifyMatchFailure(srcOp, "Already marked");
        }

        Location loc = srcOp.getLoc();

        if (srcOp.getInputs().size() != 2 || srcOp.getResults().size() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "Expects 1 input and 1 output");
        }

        Value lhs = srcOp.getInputs()[0];
        Value rhs = srcOp.getInputs()[1];
        Value output = srcOp.getResultTensors()[0];

        // Ensure inputs and output are 2D
        auto lhsType = llvm::cast<RankedTensorType>(lhs.getType());
        auto rhsType = llvm::cast<RankedTensorType>(rhs.getType());
        auto outType = llvm::cast<RankedTensorType>(output.getType());

        if (!lhsType || !rhsType || !outType || lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
            outType.getRank() != 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "Conv2DMatmulOpConversion expects 2D inputs and outputs"
            );
        }

        // Check if the Conv2D input (lhs) is produced by a CollapseShapeOp —
        // this typically means the input tensor is being flattened before the convolution.
        while (auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(lhs.getDefiningOp())) {
            lhs = extractSlice.getSource();
        }
        if (!lhs.getDefiningOp<tensor::CollapseShapeOp>() &&
            !isCollapseOrExpandShapeGeneric(lhs.getDefiningOp())) {
            return rewriter.notifyMatchFailure(srcOp, "LHS is not collapsed from 4D");
        }
        Value input = lhs.getDefiningOp()->getOperand(0);
        auto inputType = dyn_cast<RankedTensorType>(input.getType());
        if (!inputType || inputType.getRank() != 4) {
            return rewriter.notifyMatchFailure(srcOp, "Expected input to be 4D pre-collapse");
        }

        // Match transpose on weight
        if (auto transposeOp = dyn_cast<linalg::TransposeOp>(rhs.getDefiningOp())) {
            rhs = transposeOp.getInput();
        }

        // Check weights are supported
        auto weightElemType = dyn_cast<RankedTensorType>(rhs.getType()).getElementType();

        if (!weightElemType.isBF16() && !weightElemType.isInteger(8)) {
            return rewriter.notifyMatchFailure(srcOp, "Unsupported weight type");
        }

        auto weightShape = dyn_cast<ShapedType>(rhs.getType()).getShape();

        // Validate expected shape: [OC, IC] for matmul-style
        if (weightShape.size() != 2) {
            return rewriter.notifyMatchFailure(srcOp, "Expected 2D weight tensor");
        }

        // fold bias
        auto outputChannelCount = outType.getShape()[1];
        bool isInt = outType.getElementType().isInteger();
        VectorIntOrFloat bias(outputChannelCount, isInt);
        int32_t input_zp = 0;
        int32_t weightZp = 0;

        while (foldForwardPerChannelAdd(output, 1, bias, &input_zp, input, &weightZp)) {
        }

        // check if output user is expand_shape
        RankedTensorType finalType = nullptr;
        if (output.hasOneUse() && (isa<tensor::ExpandShapeOp>(*output.getUsers().begin()) ||
                                   isCollapseOrExpandShapeGeneric(*output.getUsers().begin()))) {
            output = (*output.getUsers().begin())->getResult(0);
            finalType = cast<RankedTensorType>(output.getType());
        }
        else {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected expand_shape user to determine 4D output"
            );
        }

        ScaleClampInfo scInfo = foldForwardScaleClamp(output, outputChannelCount, 20, 12);
        if (!scInfo) {
            if (isInt) {
                return rewriter.notifyMatchFailure(
                    srcOp, "Expected scale info for integer operations"
                );
            }
            scInfo = getDefaultScaleClampInfo(finalType, srcOp);
        }
        else {
            finalType = cast<RankedTensorType>(output.getType());
        }
        if (!finalType || finalType.getRank() != 4) {
            return rewriter.notifyMatchFailure(srcOp, "Expected 4D output from expand");
        }

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {input, rhs}, rewriter,
                srcOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        auto weights = rhs;
        if (weightZp != 0) {
            // Divide weights and wZp by two so we can safely add them together without overflow
            constexpr int scaleFactor = 2;
            weights = rescaleValue(weights, scaleFactor, -weightZp / 2, loc, rewriter);
            // Same for the bias
            for (auto &val : bias.ints) {
                val /= scaleFactor;
            }
            // Reduce the scaling shift to compensate
            // (We could multiply the scaling factor by 2 instead, but that could cause overflow)
            scInfo.scaleShift -= 1;
        }

        // Prepare bias (and scale for integer ops)
        auto biasScale = isInt ? createConst(interleave(bias.ints, scInfo.scaleNpu), rewriter, loc)
                               : createConst(bias.floats, rewriter, loc);

        // Compute weights
        auto weightAttr = computeConstant(weights);
        if (!weightAttr) {
            return rewriter.notifyMatchFailure(srcOp, "Failed to fold weights");
        }

        finalType = convertTypeNHWCtoNCHW(finalType);
        Value initTensor = createInitTensor(srcOp, rewriter, finalType);
        auto vectorizationMode = torq_hl::VectorizationModeEnum::None;
        input = transposeValue(input, Permutation::nhwc2nchw(), loc, rewriter);

        auto pad = rewriter.getDenseI64ArrayAttr({0, 0, 0, 0});
        auto stride = rewriter.getDenseI64ArrayAttr({1, 1});
        auto dilation = rewriter.getDenseI64ArrayAttr({1, 1});

        auto torqWeights = convertWeights(srcOp, weightAttr, rewriter);

        auto conv2dOp = rewriter.create<syna::torq_hl::Conv2DOp>(
            loc, finalType, initTensor,
            input_zp, // input_zp
            0,        // weight_zp
            scInfo.zp, scInfo.min, scInfo.max, scInfo.scaleShift,
            1,        // groups
            pad,      // pad
            stride,   // stride
            dilation, // dilation
            vectorizationMode, torqWeights, biasScale, input
        );

        auto torqOut =
            transposeValue(conv2dOp.getOutput(), Permutation::nhwc2nchw().reverse(), loc, rewriter);
        rewriter.replaceOp(output.getDefiningOp(), torqOut);

        LLVM_DEBUG({ llvm::dbgs() << "Conv2DMatmulOpConversion success\n"; });
        return success();
    }
};

struct Conv1DNcwFcwToLinalgMatmulPattern : public OpRewritePattern<linalg::Conv1DNcwFcwOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::Conv1DNcwFcwOp convOp, PatternRewriter &rewriter) const override {
        llvm::errs() << "Applying Conv1DNcwFcwToLinalgMatmul\n";
        auto loc = convOp.getLoc();

        // Extract tensors and shapes
        Value input = convOp.getInputs()[0];   // Input tensor [N,C,W]
        Value filter = convOp.getInputs()[1];  // Filter tensor [F,C,Kw]
        Value output = convOp.getOutputs()[0]; // Output tensor [N,F,Ow]

        auto inputType = cast<RankedTensorType>(input.getType());
        auto filterType = cast<RankedTensorType>(filter.getType());
        auto outputType = cast<RankedTensorType>(output.getType());

        // Extract dimensions
        ArrayRef<int64_t> inputShape = inputType.getShape();
        ArrayRef<int64_t> filterShape = filterType.getShape();
        ArrayRef<int64_t> outputShape = outputType.getShape();

        if (inputShape.size() != 3 || filterShape.size() != 3 || outputShape.size() != 3) {
            return rewriter.notifyMatchFailure(convOp, "Expected 3D tensors for Conv1D");
        }

        // Extract convolution parameters
        SmallVector<int64_t> strides = llvm::to_vector<4>(
            llvm::map_range(convOp.getStrides(), [](APInt v) { return v.getSExtValue(); })
        );
        SmallVector<int64_t> dilations = llvm::to_vector<4>(
            llvm::map_range(convOp.getDilations(), [](APInt v) { return v.getSExtValue(); })
        );

        int64_t N = inputShape[0];       // Batch size
        int64_t C = inputShape[1];       // Input channels
        int64_t F = filterShape[0];      // Output channels/filters
        int64_t Kw = filterShape[2];     // Kernel width
        int64_t Ow = outputShape[2];     // Output width
        int64_t stride = strides[0];     // Stride value
        int64_t dilation = dilations[0]; // Dilation value

        // Step 1: Unfold the input tensor using im2col approach
        // Each position in the output corresponds to a patch of the input
        auto elemType = inputType.getElementType();
        auto outputElemType = outputType.getElementType();
        // Create a tensor to hold the unfolded input
        // Shape: [Ow, C*Kw] - each row contains a full patch for one output position
        SmallVector<int64_t> unfoldedShape = {Ow, C * Kw};
        auto unfoldedType = RankedTensorType::get(unfoldedShape, elemType);
        auto unfoldedInit = rewriter.create<tensor::EmptyOp>(loc, unfoldedShape, elemType);

        // Create the im2col transformation using a linalg.generic
        SmallVector<AffineExpr> unfoldIndexExprs;
        auto dim0 = rewriter.getAffineDimExpr(0); // Output position (Ow dimension)
        auto dim1 = rewriter.getAffineDimExpr(1); // Input channel and kernel position

        // dim1 / Kw gives us the channel index
        auto channelIdx = dim1.floorDiv(rewriter.getAffineConstantExpr(Kw));
        // dim1 % Kw gives us the kernel position
        auto kernelIdx = dim1 % rewriter.getAffineConstantExpr(Kw);
        // Calculate input position: outputPos * stride + kernelIdx * dilation
        auto inputPosExpr = dim0 * rewriter.getAffineConstantExpr(stride) +
                            kernelIdx * rewriter.getAffineConstantExpr(dilation);

        unfoldIndexExprs.push_back(rewriter.getAffineConstantExpr(0)); // N dimension (batch)
        unfoldIndexExprs.push_back(channelIdx);                        // C dimension (channels)
        unfoldIndexExprs.push_back(inputPosExpr);                      // W dimension (width)

        auto unfoldIndexMap = AffineMap::get(2, 0, unfoldIndexExprs, rewriter.getContext());
        auto outputIndexMap = AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());

        // Create the generic op for unfolding with explicit iterator types
        SmallVector<utils::IteratorType> iteratorTypes(2, utils::IteratorType::parallel);

        auto im2col = rewriter.create<linalg::GenericOp>(
            loc, TypeRange{unfoldedType}, ValueRange{input}, ValueRange{unfoldedInit},
            ArrayRef<AffineMap>{unfoldIndexMap, outputIndexMap}, iteratorTypes,
            [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
                nestedBuilder.create<linalg::YieldOp>(nestedLoc, blockArgs[0]);
            }
        );

        // Set torq.im2col attribute so that we can easily recognize this op during tiling
        im2col->setAttr("torq.im2col", rewriter.getBoolAttr(true));
        auto unfoldedInput = im2col.getResult(0);

        // Step 2: Reshape the filter tensor from [F, C, Kw] to [F, C*Kw]
        SmallVector<int64_t> reshapedFilterShape = {F, C * Kw};
        auto reshapedFilterType =
            RankedTensorType::get(reshapedFilterShape, filterType.getElementType());
        auto reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
            loc, reshapedFilterType, filter, ArrayRef<ReassociationIndices>{{0}, {1, 2}}
        );

        // Step 3: Create the matmul operation
        // We'll do: [F, C*Kw] @ [Ow, C*Kw]^T -> [F, Ow]
        // First, we need to transpose the unfolded input
        SmallVector<int64_t> transposedUnfoldedShape = {C * Kw, Ow};
        // auto transposedUnfoldedType = RankedTensorType::get(transposedUnfoldedShape, elemType);
        auto transposedUnfoldedInit =
            rewriter.create<tensor::EmptyOp>(loc, transposedUnfoldedShape, elemType);

        auto transposedUnfolded = rewriter.create<linalg::TransposeOp>(
            loc, unfoldedInput, transposedUnfoldedInit, ArrayRef<int64_t>{1, 0}
        );

        // Create the matmul output tensor [F, Ow]
        SmallVector<int64_t> matmulResultShape = {F, Ow};
        auto matmulResultType = RankedTensorType::get(matmulResultShape, outputElemType);
        auto matmulInit = rewriter.create<tensor::EmptyOp>(loc, matmulResultShape, outputElemType);

        // Perform the actual matmul
        // Perform the actual matmul
        SmallVector<Value> inputs;
        inputs.push_back(reshapedFilter.getResult());
        inputs.push_back(transposedUnfolded.getResults()[0]);

        SmallVector<Value> outputs;
        outputs.push_back(matmulInit.getResult());

        auto matmulOp =
            rewriter.create<linalg::MatmulOp>(loc, TypeRange{matmulResultType}, inputs, outputs);

        // Step 4: Reshape the result back to [N, F, Ow]
        if (N == 1) {
            // Simply reshape to add the batch dimension
            auto finalResult = rewriter.create<tensor::ExpandShapeOp>(
                loc, matmulResultType, matmulOp.getResults()[0],
                ArrayRef<ReassociationIndices>{{0, 1}, {2}}
            );

            rewriter.replaceOp(convOp, finalResult);
        }
        else {
            return rewriter.notifyMatchFailure(
                convOp, "Batched Conv1D not supported in this pattern"
            );
        }

        return success();
    }
};

struct Conv1DNcwFcwToLinalgConv2DPattern : public OpRewritePattern<linalg::Conv1DNcwFcwOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::Conv1DNcwFcwOp convOp, PatternRewriter &rewriter) const override {
        llvm::errs() << "Applying Conv1DNcwFcwToLinalgConv2D\n";
        auto loc = convOp.getLoc();

        // Get operands
        Value input = convOp.getInputs()[0];
        Value filter = convOp.getInputs()[1];
        Value output = convOp.getOutputs()[0];

        // Get types and shapes
        auto inputType = cast<RankedTensorType>(input.getType());
        auto filterType = cast<RankedTensorType>(filter.getType());
        auto outputType = cast<RankedTensorType>(output.getType());

        // Add height dimension (1) to input: [N,C,W] -> [N,C,1,W]
        // Need to use proper reassociation indices
        SmallVector<ReassociationIndices> inputReassoc = {{0}, {1}, {2, 3}};
        auto expandedInputType = RankedTensorType::get(
            {inputType.getShape()[0], inputType.getShape()[1], 1, inputType.getShape()[2]},
            inputType.getElementType()
        );

        auto expandedInput =
            rewriter.create<tensor::ExpandShapeOp>(loc, expandedInputType, input, inputReassoc);

        // Transpose to NHWC format: [N,C,1,W] -> [N,1,W,C]
        SmallVector<int64_t> inputPerm = {0, 2, 3, 1};
        Value nhwcInput = transposeValue(expandedInput, inputPerm, loc, rewriter);

        // Add height dimension to filter: [F,C,W] -> [F,C,1,W]
        SmallVector<ReassociationIndices> filterReassoc = {{0}, {1}, {2, 3}};
        auto expandedFilterType = RankedTensorType::get(
            {filterType.getShape()[0], filterType.getShape()[1], 1, filterType.getShape()[2]},
            filterType.getElementType()
        );

        auto expandedFilter =
            rewriter.create<tensor::ExpandShapeOp>(loc, expandedFilterType, filter, filterReassoc);

        // Transpose to HWCF format: [F,C,1,W] -> [1,W,C,F]
        SmallVector<int64_t> filterPerm = {2, 3, 1, 0};
        Value hwcfFilter = transposeValue(expandedFilter, filterPerm, loc, rewriter);

        // Add height dimension to output: [N,F,W] -> [N,F,1,W]
        SmallVector<ReassociationIndices> outputReassoc = {{0}, {1}, {2, 3}};
        auto expandedOutputType = RankedTensorType::get(
            {outputType.getShape()[0], outputType.getShape()[1], 1, outputType.getShape()[2]},
            outputType.getElementType()
        );

        auto expandedOutput =
            rewriter.create<tensor::ExpandShapeOp>(loc, expandedOutputType, output, outputReassoc);

        // Transpose to NHWC format: [N,F,1,W] -> [N,1,W,F]
        SmallVector<int64_t> outputPerm = {0, 2, 3, 1};
        Value nhwcOutput = transposeValue(expandedOutput, outputPerm, loc, rewriter);

        // Get attributes
        auto stridesAttr = convOp.getStrides();
        auto dilationsAttr = convOp.getDilations();

        // Convert 1D strides/dilations to 2D (add height dimension)
        SmallVector<int64_t> strides2d = {1};
        strides2d.push_back(stridesAttr.getValues<int64_t>()[0]);
        SmallVector<int64_t> dilations2d = {1};
        dilations2d.push_back(dilationsAttr.getValues<int64_t>()[0]);

        auto attrType = RankedTensorType::get({2}, rewriter.getIntegerType(64));
        auto stridesAttr2d = DenseIntElementsAttr::get(attrType, strides2d);
        auto dilationsAttr2d = DenseIntElementsAttr::get(attrType, dilations2d);

        // Create Conv2D
        auto conv2d = rewriter.create<linalg::Conv2DNhwcHwcfOp>(
            loc, nhwcOutput.getType(), ValueRange{nhwcInput, hwcfFilter}, ValueRange{nhwcOutput},
            stridesAttr2d, dilationsAttr2d
        );

        // Transpose result back: [N,1,W,F] -> [N,F,1,W]
        Value transposedResult = transposeValue(conv2d.getResult(0), {0, 3, 1, 2}, loc, rewriter);

        // Collapse height dimension: [N,F,1,W] -> [N,F,W]
        auto collapsedResult = rewriter.create<tensor::CollapseShapeOp>(
            loc, outputType, transposedResult, outputReassoc
        );

        rewriter.replaceOp(convOp, collapsedResult.getResult());
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
        auto empty = rewriter.create<tensor::EmptyOp>(loc, scalarTy, ValueRange{});
        Value scalar0D = rewriter.create<tensor::InsertOp>(loc, negZpI16, empty, ValueRange{});

        // FIXME: we don't support tensor-scalar arith ops when scalar is non constant
        // so we create a tensor for the scalar with the same shape as the input tensor.
        // This is not efficient but should be ok for now as the scalar is expected to be
        // a zero point.
        Value cEmpty = rewriter.create<tensor::EmptyOp>(loc, tTy.getShape(), i16);

        SmallVector<int64_t> dim(tTy.getRank());
        for (int i = 0; i < tTy.getRank(); i++) {
            dim[i] = i;
        }
        Value cInit =
            rewriter.create<linalg::BroadcastOp>(loc, scalar0D, cEmpty, dim).getResults()[0];

        // Identity for the tensor, scalar map ()->() for the 0-D input, identity for the output.
        AffineMap id = rewriter.getMultiDimIdentityMap(tTy.getRank());
        SmallVector<utils::IteratorType> iters(tTy.getRank(), utils::IteratorType::parallel);

        Value out = rewriter.create<tensor::EmptyOp>(loc, tTy.getShape(), tTy.getElementType());
        auto g = rewriter.create<linalg::GenericOp>(
            loc, TypeRange{tTy}, ValueRange{tensorI16, cInit}, ValueRange{out},
            ArrayRef<AffineMap>{id, id, id}, iters,
            [&](OpBuilder &b, Location l, ValueRange args) {
                // TODO: double-check if zeropoint must be substracted or added using llvm-cpu with
                // a given input matrix
                Value sum = b.create<arith::SubIOp>(l, args[0], args[1]);
                b.create<linalg::YieldOp>(l, sum);
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
            rewriter.create<arith::ExtSIOp>(loc, RankedTensorType::get(aTy.getShape(), i16), aI8);
        Value bI16 =
            rewriter.create<arith::ExtSIOp>(loc, RankedTensorType::get(bTy.getShape(), i16), bI8);

        auto zpL16 = rewriter.create<arith::TruncIOp>(loc, i16, zpLRaw); // i8/i16 -> i16
        auto zpR16 = rewriter.create<arith::TruncIOp>(loc, i16, zpRRaw); // i8/i16 -> i16

        Value aAdj = addScalar0DWithGeneric(rewriter, loc, aI16, zpL16);
        Value bAdj = addScalar0DWithGeneric(rewriter, loc, bI16, zpR16);

        // Zero i32 accumulator and (batch_)matmul i16×i16 → i32.
        Value cEmpty =
            rewriter.create<tensor::EmptyOp>(loc, outTy.getShape(), outTy.getElementType());
        Value z32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
        Value cInit =
            rewriter.create<linalg::FillOp>(loc, ValueRange{z32}, ValueRange{cEmpty}).result();

        Value result;
        if (outTy.getRank() == 3) {
            result = rewriter
                         .create<linalg::BatchMatmulOp>(
                             loc, TypeRange{outTy}, ValueRange{aAdj, bAdj}, ValueRange{cInit}
                         )
                         .getResult(0);
        }
        else {
            result = rewriter
                         .create<linalg::MatmulOp>(
                             loc, TypeRange{outTy}, ValueRange{aAdj, bAdj}, ValueRange{cInit}
                         )
                         .getResult(0);
        }

        rewriter.replaceOp(op, result);
        return success();
    }
};

// Checker methods for convolutions with input: NHWC, weights: HWC(F)
struct Check {
    static constexpr int ih = 1, kh = 0;
    static constexpr int iw = ih + 1, kw = kh + 1;
    static constexpr int maxKerHW = 9;
    using Shape = ArrayRef<int64_t>;

    // Check that the kernel shape is small enough
    static bool isKerSmall(Shape iShape, Shape wShape, Shape padShape) {
        return iShape.size() == 4 && wShape.size() >= 3 && wShape[kh] <= maxKerHW &&
               wShape[kw] <= maxKerHW;
    }

    // Check that the kernel shape is equal to the input shape (without padding)
    static bool isKerEqInput(Shape iShape, Shape wShape, Shape padShape) {
        bool noPadding = llvm::all_of(padShape, [](auto p) { return p == 0; });
        return noPadding && iShape.size() == 4 && wShape.size() >= 3 && wShape[kh] > 1 &&
               wShape[kw] > 1 && iShape[ih] == wShape[kh] && iShape[iw] == wShape[kw];
    }
};

void populateLinalgToTorqHLPrePatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<BfloatGenericErfPattern>(context);
    patterns.insert<BfloatErfPattern>(context);
    patterns.insert<BfloatGenericTanhPattern>(context);
    patterns.insert<BfloatTanhPattern>(context);
    patterns.insert<BfloatSoftmaxPattern>(context);
    patterns.insert<BfloatDivfPattern>(context);
    patterns.insert<BfloatReciprocalPattern>(context);
    patterns.insert<TensorBitcastPattern>(context);
    patterns.insert<ArithOnTensorToLinalgPattern<arith::AndIOp>>(context);
    patterns.insert<ArithOnTensorToLinalgPattern<arith::CmpIOp>>(context);
    patterns.insert<ArithOnTensorToLinalgPattern<arith::MulIOp>>(context);
    patterns.insert<ArithOnTensorToLinalgPattern<arith::OrIOp>>(context);
    patterns.insert<ArithOnTensorToLinalgPattern<arith::AddIOp>>(context);
    patterns.insert<ArithOnTensorToLinalgPattern<arith::ExtSIOp>>(context);

    patterns.insert<Conv2dConvert<linalg::Conv2DNhwcHwcfOp, syna::torq_hl::Conv2DOp>>(
        context, 3, Permutation::nhwc2nchw(), Permutation::hwcf2fchw(), 28, 12, Check::isKerSmall,
        markFuseGroups
    );
    patterns.insert<Conv2dConvert<linalg::DepthwiseConv2DNhwcHwcOp, torq_hl::DepthwiseConv2DOp>>(
        context, 3, Permutation::nhwc2nchw(), Permutation::hwc2chw(), 20, 12, Check::isKerSmall,
        markFuseGroups
    );
    patterns.insert<Conv2dConvert<linalg::DepthwiseConv2DNhwcHwcOp, torq_hl::DepthwiseConv2DOp>>(
        context, 3, Permutation::none(), Permutation::none(), 20, 12, Check::isKerEqInput,
        markFuseGroups
    );

    patterns.insert<PoolingNhwcMaxOpConversion>(context, markFuseGroups);

    patterns.insert<FCMatmulOpConversion>(context, markFuseGroups);
    patterns.insert<Conv2DMatmulOpConversion>(context, markFuseGroups);
    if (clConv1dAsMatmul) {
        patterns.insert<Conv1DNcwFcwToLinalgMatmulPattern>(context);
    }
    else {
        patterns.insert<Conv1DNcwFcwToLinalgConv2DPattern>(context);
    }

    if (!markFuseGroups) {
        patterns.insert<QuantizedBatchMatmulPattern>(context);
    }
}

} // namespace mlir::syna::torq
