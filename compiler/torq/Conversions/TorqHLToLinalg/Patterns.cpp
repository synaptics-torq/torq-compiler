// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"

namespace mlir::syna::torq {

struct TorqHLTransposeToLinalgPattern : public OpRewritePattern<torq_hl::TransposeOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hl::TransposeOp op, PatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
            op, op.getInput(), op.getInit(), op.getPermAttr()
        );
        return success();
    }
};

void populateTorqHLToLinalgPatterns(
    MLIRContext *context, RewritePatternSet &patterns, TypeConverter &typeConverter
) {
    patterns.add<TorqHLTransposeToLinalgPattern>(context);
}

} // namespace mlir::syna::torq
