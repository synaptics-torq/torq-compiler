// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-optimize-pow-op-pattern"

namespace mlir::syna::torq {

// Rewrites math.powf(x, k) inside an elementwise linalg.generic into a simpler
// op the torq backend can lower:
//   powf(x, 0.5) -> sqrt(x)
// The generic in the failing case has a single tensor input and a body-local
// constant exponent: `%c = arith.constant 0.5; math.powf %in, %c`, so the
// rewrite is a straight in-body op replacement. Downstream sqrt lowering
// (rsqrt Newton-Raphson + mul) then fires in the same greedy round.
class PowConstantExponentPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {
        // Expect exactly one real op in the body (plus the yield).
        if (srcOp.getBody()->getOperations().size() != 2) {
            return rewriter.notifyMatchFailure(srcOp, "expected single elementwise op in body");
        }

        auto powOp = dyn_cast<math::PowFOp>(&srcOp.getBody()->front());
        if (!powOp) {
            return rewriter.notifyMatchFailure(srcOp, "body op is not math.powf");
        }

        // Exponent must be a constant float.
        APFloat exp(0.0);
        if (!matchPattern(powOp.getRhs(), m_ConstantFloat(&exp))) {
            return rewriter.notifyMatchFailure(srcOp, "powf exponent is not a constant");
        }

        // powf(x, 0.5) -> sqrt(x)
        if (exp.isExactlyValue(0.5)) {
            rewriter.replaceOpWithNewOp<math::SqrtOp>(powOp, powOp.getLhs());
            return success();
        }

        // TODO: Extend here for other exponents as needed
        return rewriter.notifyMatchFailure(srcOp, "unhandled powf exponent");
    }
};

void populateOptimizePowPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.add<PowConstantExponentPattern>(context);
}

} // namespace mlir::syna::torq