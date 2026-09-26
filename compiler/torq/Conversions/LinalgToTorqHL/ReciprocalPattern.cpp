// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// ReciprocalOpPattern
// -------------------
// Lower a bf16 linalg.reciprocal directly to the torq_hl LUT pipeline:
//
//   recip bits = (0x7E80 - x) + mantissaLUT[x & 0xFF]        (i16 wrap-around)
//
// The mantissa LUT depends only on the low byte of the i16 bit pattern x, so
// the table slice reads it exactly: with the ACT rescale (bias=0, scale=256),
// tc = s16(256 * x) is a multiple of 128 (zero 7-bit fraction) and
//   tx = 2 * ((x & 0xFF) + 128) mod 512
// so the 256 logical entries are placed on the even slots of the HW's
// 512-entry table (odd slots hold 0 and are never read). This is a single
// torq_hl.table on the full i16 bit pattern — no trunci/exti slices.
//
// The recombine steps (0x7E80 - x and lut + expo) are left as linalg.generics
// for the existing AddOpPattern in this same conversion, which lowers them to
// torq_hl.add exactly as in the decomposed pipeline.
//
// The --torq-enable-reciprocal-inf variant is decomposed earlier (see
// BfloatReciprocalPattern in DecomposeLinalgOpsPattern.cpp) and never
// reaches this pattern.

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-torq-reciprocal"

namespace mlir::syna::torq {

namespace {

class ReciprocalOpPattern : public OpRewritePattern<linalg::ReciprocalOp> {
  public:
    ReciprocalOpPattern(MLIRContext *context)
        : OpRewritePattern<linalg::ReciprocalOp>(context, /*benefit=*/10) {
        setDebugName("ReciprocalOpPattern");
    }

    LogicalResult
    matchAndRewrite(linalg::ReciprocalOp op, PatternRewriter &rewriter) const override {

        // checks
        if (op.getInputs().size() != 1) {
            return rewriter.notifyMatchFailure(op, "Expected exactly one input");
        }
        if (op.getOutputs().size() != 1) {
            return rewriter.notifyMatchFailure(op, "Expected exactly one output");
        }
        auto inType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
        auto outType = dyn_cast<RankedTensorType>(op.getType(0));
        if (!inType || !outType) {
            return rewriter.notifyMatchFailure(op, "Expected ranked tensors");
        }
        if (!outType.getElementType().isBF16() ||
            inType.getElementType() != outType.getElementType()) {
            return rewriter.notifyMatchFailure(op, "Expected bf16 reciprocal");
        }
        // If the user forced this op onto Host/CSS, keep it as linalg for the
        // CPU path instead of converting it.
        auto [isForced, forced] = setTargetExecutorIfForced(op, rewriter, "reciprocal");
        if (isForced) {
            return forced;
        }

        auto loc = op.getLoc();
        auto ctx = rewriter.getContext();
        auto shape = outType.getShape();
        auto rank = (size_t)outType.getRank();
        auto i16 = rewriter.getI16Type();
        auto i16TType = RankedTensorType::get(shape, i16);

        // 1) bitcast the input to i16 (torq_hl.identity)
        auto x = makeBitcast(op, rewriter, i16TType, op.getInputs()[0]);

        // 2) mantissa LUT on the full i16 bit pattern (torq_hl.table {0, 256})
        auto lutData = getBfloatReciprocalLutData();
        // packed word = (slope << 16) | (base & 0xFFFF); slope is 0 here and
        // the base sits on the even entries (odd entries are never read).
        SmallVector<int32_t> lutWords(512, 0);
        for (int i = 0; i < 256; i++) {
            lutWords[2 * i] = static_cast<int32_t>(static_cast<uint8_t>(lutData[i]));
        }
        auto lutVal16 =
            makeI16LUTFromVals(op, rewriter, x, lutWords, /*lutBias=*/0, /*lutScale=*/256);

        // 3) exponent term: 0x7E80 - x (lowered by AddOpPattern to torq_hl.add)
        auto cExpo = arith::ConstantOp::create(
            rewriter, loc, rewriter.getIntegerAttr(i16, kBfloatReciprocalExpoConst)
        );
        auto computedExpo =
            linalg::GenericOp::create(
                rewriter, loc, TypeRange{i16TType}, ValueRange{x},
                ValueRange{tensor::EmptyOp::create(rewriter, loc, i16TType, ValueRange{})},
                SmallVector<AffineMap>(2, AffineMap::getMultiDimIdentityMap(rank, ctx)),
                SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                [&](OpBuilder &b, Location l, ValueRange args) {
                    linalg::YieldOp::create(
                        b, l, ValueRange{arith::SubIOp::create(b, l, cExpo, args[0])}
                    );
                }
            ).getResult(0);

        // 4) recombine: lutVal16 + computedExpo (lowered by AddOpPattern)
        auto computed =
            linalg::GenericOp::create(
                rewriter, loc, TypeRange{i16TType}, ValueRange{lutVal16, computedExpo},
                ValueRange{tensor::EmptyOp::create(rewriter, loc, i16TType, ValueRange{})},
                SmallVector<AffineMap>(3, AffineMap::getMultiDimIdentityMap(rank, ctx)),
                SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                [&](OpBuilder &b, Location l, ValueRange args) {
                    linalg::YieldOp::create(
                        b, l, ValueRange{arith::AddIOp::create(b, l, args[0], args[1])}
                    );
                }
            ).getResult(0);

        // 5) bitcast the result back to bf16 (torq_hl.identity)
        rewriter.replaceOp(op, makeBitcast(op, rewriter, outType, computed));
        return success();
    }
};

} // namespace

void populateReciprocalPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.insert<ReciprocalOpPattern>(context);
}

} // namespace mlir::syna::torq
