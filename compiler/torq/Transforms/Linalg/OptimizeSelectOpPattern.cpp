// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Utils/ShapeUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "numeric"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-optimize-select-op-pattern"

namespace mlir::syna::torq {

static arith::SelectOp getSelectOp(linalg::GenericOp op, bool allowConstants) {
    // Expect 3 inputs, 1 init, 1 result.
    if (op.getNumDpsInputs() != 3 || op.getNumDpsInits() != 1 || op.getNumResults() != 1) {
        LLVM_DEBUG({ llvm::dbgs() << "SelectOp: expected 3 ins / 1 init / 1 result\n"; });
        return {};
    }

    // Elementwise: all loops parallel (or scalar rank=0 with 0 loops).
    auto outTy = cast<RankedTensorType>(op.getResultTensors()[0].getType());
    if (outTy.getRank() > 0 && op.getNumLoops() < 1) {
        LLVM_DEBUG({ llvm::dbgs() << "SelectOp: expected loops for ranked output\n"; });
        return {};
    }
    if (op.getNumParallelLoops() != op.getNumLoops()) {
        LLVM_DEBUG({ llvm::dbgs() << "SelectOp: expected all loops parallel\n"; });
        return {};
    }

    // All DPS inputs must be used in the payload.
    for (int i = 0; i < 3; ++i) {
        if (!op.payloadUsesValueFromOperand(op.getDpsInputOperand(i))) {
            LLVM_DEBUG({ llvm::dbgs() << "SelectOp: input " << i << " not used\n"; });
            return {};
        }
    }

    auto *body = op.getBody();
    if (!body) {
        LLVM_DEBUG({ llvm::dbgs() << "SelectOp: missing body\n"; });
        return {};
    }

    auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
        LLVM_DEBUG({ llvm::dbgs() << "SelectOp: expected single yield\n"; });
        return {};
    }

    auto selectOp = yieldOp.getOperand(0).getDefiningOp<arith::SelectOp>();
    if (!selectOp) {
        LLVM_DEBUG({ llvm::dbgs() << "SelectOp: yield is not produced by arith.select\n"; });
        return {};
    }

    // In a linalg.generic, block args are: ins..., outs...
    // Here: args[0..2] are inputs, args[3] is the init/output.
    Block &block = *body;
    if (block.getNumArguments() != 4) {
        LLVM_DEBUG({ llvm::dbgs() << "SelectOp: expected 4 block args (3 ins + 1 out)\n"; });
        return {};
    }

    auto isOkOperand = [&](Value v, int inputIdx) -> bool {
        // Optionally peel integer casts that upstream might introduce.
        if (auto extsi = v.getDefiningOp<arith::ExtSIOp>())
            v = extsi.getIn();
        if (auto extui = v.getDefiningOp<arith::ExtUIOp>())
            v = extui.getIn();
        if (auto trunci = v.getDefiningOp<arith::TruncIOp>())
            v = trunci.getIn();

        // Allow the corresponding input block argument, or a constant if enabled.
        if (v == block.getArgument(inputIdx))
            return true;
        if (allowConstants && isa_and_nonnull<arith::ConstantOp>(v.getDefiningOp()))
            return true;

        // Explicitly reject using the init/output block argument.
        if (v == block.getArgument(3))
            return false;

        return false;
    };

    if (!isOkOperand(selectOp.getCondition(), 0) || !isOkOperand(selectOp.getTrueValue(), 1) ||
        !isOkOperand(selectOp.getFalseValue(), 2)) {
        LLVM_DEBUG({
            llvm::dbgs() << "SelectOp: operands not from expected block args/constants\n";
        });
        return {};
    }

    return selectOp;
}

// select op with 3 inputs, 1 output
class BroadcastSelectOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {

        if (srcOp->getAttrOfType<BoolAttr>("broadcasted")) {
            return rewriter.notifyMatchFailure(
                srcOp, "select op broadcast pattern skipping op with `broadcasted` attribute\n"
            );
        }

        if (srcOp.getNumDpsInputs() != 3) {
            return rewriter.notifyMatchFailure(
                srcOp, "select op broadcast pattern expects 3 DPS inputs\n"
            );
        }

        if (srcOp.getNumDpsInits() != 1 || srcOp.getNumResults() != 1) {
            return rewriter.notifyMatchFailure(
                srcOp, "select op broadcast pattern expects 1 DPS init / 1 result\n"
            );
        }

        Value cond = srcOp.getInputs()[0];
        Value tVal = srcOp.getInputs()[1];
        Value fVal = srcOp.getInputs()[2];

        auto condType = dyn_cast<RankedTensorType>(cond.getType());
        auto tType = dyn_cast<RankedTensorType>(tVal.getType());
        auto fType = dyn_cast<RankedTensorType>(fVal.getType());
        auto resType = dyn_cast<RankedTensorType>(srcOp.getResultTypes().front());

        if (!condType || !tType || !fType || !resType) {
            return rewriter.notifyMatchFailure(
                srcOp,
                "select op broadcast pattern expects ranked tensor type for inputs and result\n"
            );
        }

        if (!condType.getElementType().isInteger(1)) {
            return rewriter.notifyMatchFailure(
                srcOp, "select op broadcast pattern expects i1 element type for condition\n"
            );
        }

        Type tDtype = tType.getElementType();
        Type fDtype = fType.getElementType();
        Type resDtype = resType.getElementType();

        if (tDtype != fDtype || tDtype != resDtype) {
            return rewriter.notifyMatchFailure(
                srcOp,
                "select op broadcast pattern expects true/false/result element types to match\n"
            );
        }

        if (resDtype.isF64()) {
            return rewriter.notifyMatchFailure(
                srcOp, "select op broadcast pattern does not support f64\n"
            );
        }

        auto selectOp = getSelectOp(srcOp, true);
        if (!selectOp) {
            return rewriter.notifyMatchFailure(srcOp, "generic op is not expected select op\n");
        }

        SmallVector<Value> inputs(srcOp.getInputs().begin(), srcOp.getInputs().end());

        auto isConstantValue = [](Value val) {
            if (auto definingOp = val.getDefiningOp()) {
                return isa<arith::ConstantOp>(definingOp);
            }
            return false;
        };

        for (auto it : llvm::enumerate(inputs)) {
            auto inp = it.value();
            auto inpType = dyn_cast<RankedTensorType>(inp.getType());
            auto inpRank = inpType.getRank();

            /* Compiler constraints */
            if (inpRank == 0) {
                return rewriter.notifyMatchFailure(
                    srcOp, "select op broadcast pattern does not support 0-rank inputs\n"
                );
            }
            if (isConstantValue(inp) && inpRank < 2) {
                return rewriter.notifyMatchFailure(
                    srcOp,
                    "select op broadcast pattern skipping op with constant input with rank=1\n"
                );
            }
        }

        if (failed(broadcastInputs(srcOp, inputs, rewriter))) {
            return rewriter.notifyMatchFailure(
                srcOp, "select op broadcast pattern failed to broadcast inputs\n"
            );
        }

        AffineMap m = srcOp.getMatchingIndexingMap(srcOp.getDpsInitOperand(0));
        SmallVector<AffineMap> newIndexingMaps(inputs.size() + srcOp.getNumDpsInits(), m);

        // update the inputs and related attr of srcOp
        rewriter.modifyOpInPlace(srcOp, [&]() {
            for (auto it : llvm::enumerate(inputs)) {
                srcOp->setOperand(it.index(), it.value());
            }
            srcOp.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(newIndexingMaps));
        });

        srcOp->setAttr("broadcasted", BoolAttr::get(srcOp->getContext(), true));

        return success();
    }
};

void populateOptimizeSelectPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.add<PromoteScalarsTo1D>(context);
    patterns.add<BroadcastSelectOpPattern>(context);
    patterns.add<ReshapeToCollapseExpand>(context);
    tensor::populateReassociativeReshapeFoldingPatterns(patterns);
    patterns.add<ComposeExpandOfCollapseOp<tensor::ExpandShapeOp, tensor::CollapseShapeOp>>(context
    );
}

} // namespace mlir::syna::torq
