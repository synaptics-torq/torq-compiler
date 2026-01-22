// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Conversions/LinalgToTorqHL/MatchingFunctions.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ComputeConstants.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/TorqUtils.h"

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

#define DEBUG_TYPE "torq-optimize-arith-elementwise-binary-op-pattern"

namespace mlir::syna::torq {

// Optimize elementwise binary ops so that if one of the operands is a constant scalar,
// we create a linalg.generic op that takes a tensor constant as input instead of
// a scalar constant. This is to enable further optimizations downstream.
// TODO: whenever the ElementwiseBinaryOp kernel supports broadcasting, we can remove
// this pass altogether.
class ArithElementwiseBinaryOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const {
        if (srcOp->getAttrOfType<BoolAttr>("arithCmpOptimized")) {
            return failure();
        }

        if (srcOp.getNumDpsInputs() != 1) {
            return rewriter.notifyMatchFailure(
                srcOp, "ArithElementBinaryOpPattern only supports 1 input"
            );
        }

        // We only consider the case where there is a single elementwise binary op here.
        // (second one is the yield op)
        if (srcOp.getBody()->getOperations().size() != 2) {
            return rewriter.notifyMatchFailure(srcOp, "linalg.generic expects 1 valid op");
        }

        Operation *op = getElementwiseBinaryOp(srcOp, /*allowCmp=*/true);
        if (!op) {
            return rewriter.notifyMatchFailure(srcOp, "Not an elementwise binary op");
        }

        // TODO: support cases where first operand is constant and second is input tensor
        auto resultType = mlir::dyn_cast<RankedTensorType>(srcOp.getResult(0).getType());
        if (!resultType) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected the output type to be a RankedTensorType"
            );
        }

        if (!isa<arith::CmpFOp>(op) && !isa<arith::CmpIOp>(op) && !isa<arith::AndIOp>(op) &&
            !isa<arith::OrIOp>(op) && !isa<arith::XOrIOp>(op) && !isa<arith::MinimumFOp>(op) &&
            !isa<arith::MaximumFOp>(op) && !isa<arith::MinNumFOp>(op) &&
            !isa<arith::MaxNumFOp>(op) && !isa<arith::MinSIOp>(op) && !isa<arith::MaxSIOp>(op)) {
            return rewriter.notifyMatchFailure(srcOp, "Expected arith elementwise binary op");
        }

        // We don't support logical not op here even though it is implemented with a binary XOR
        // operation, since it is a unary operation in practice.
        std::string failReason;
        if (isLogicNotOp(op, failReason)) {
            return rewriter.notifyMatchFailure(
                srcOp, "unary XOrIOp(logical not) op can't be considered as binary op"
            );
        }

        auto input = srcOp.getInputs()[0];
        auto inputType = mlir::cast<RankedTensorType>(input.getType());

        auto lhs = op->getOperand(0);
        auto rhs = op->getOperand(1);

        Value newInput0, newInput1;

        if (auto rhsConstOp = dyn_cast_if_present<arith::ConstantOp>(rhs.getDefiningOp())) {
            // Create a constant tensor from the scalar constant value.
            Attribute constAttr = computeConstant(srcOp, false);
            if (!constAttr) {
                constAttr = rhsConstOp.getValue();
            }
            if (!constAttr) {
                return failure();
            }
            auto constTensor = rewriter.create<arith::ConstantOp>(
                srcOp.getLoc(), inputType, DenseElementsAttr::get(inputType, constAttr)
            );

            newInput0 = input;
            newInput1 = constTensor;
        }
        else if (auto lhsConstOp = dyn_cast_if_present<arith::ConstantOp>(lhs.getDefiningOp())) {
            Attribute constAttr = computeConstant(srcOp, false);
            if (!constAttr) {
                constAttr = lhsConstOp.getValue();
            }
            if (!constAttr) {
                return failure();
            }
            auto constTensor = rewriter.create<arith::ConstantOp>(
                srcOp.getLoc(), inputType, DenseElementsAttr::get(inputType, constAttr)
            );

            newInput0 = constTensor;
            newInput1 = input;
        }
        else {
            return failure();
        }

        if (!newInput0 || !newInput1) {
            return failure();
        }

        auto loc = srcOp->getLoc();
        unsigned tensorRank = inputType.getRank();

        SmallVector<AffineMap, 3> maps;
        for (unsigned i = 0; i < 3; ++i) {
            maps.push_back(AffineMap::getMultiDimIdentityMap(tensorRank, rewriter.getContext()));
        }

        SmallVector<utils::IteratorType> iteratorTypes(tensorRank, utils::IteratorType::parallel);

        SmallVector<Value, 2> inputs{newInput0, newInput1};
        auto genericOp = rewriter.create<linalg::GenericOp>(
            loc, srcOp.getResultTypes(), inputs, srcOp.getOutputs(), maps, iteratorTypes,
            [&](OpBuilder &b, Location innerLoc, ValueRange args) {
                Value res;
                if (auto cmpf = dyn_cast<arith::CmpFOp>(op)) {
                    res = b.create<arith::CmpFOp>(innerLoc, cmpf.getPredicate(), args[0], args[1]);
                }
                else if (auto cmpi = dyn_cast<arith::CmpIOp>(op)) {
                    res = b.create<arith::CmpIOp>(innerLoc, cmpi.getPredicate(), args[0], args[1]);
                }
                else if (auto andi = dyn_cast<arith::AndIOp>(op)) {
                    res = b.create<arith::AndIOp>(innerLoc, args[0], args[1]);
                }
                else if (auto ori = dyn_cast<arith::OrIOp>(op)) {
                    res = b.create<arith::OrIOp>(innerLoc, args[0], args[1]);
                }
                else if (auto xori = dyn_cast<arith::XOrIOp>(op)) {
                    res = b.create<arith::XOrIOp>(innerLoc, args[0], args[1]);
                }
                else if (auto minf = dyn_cast<arith::MinimumFOp>(op)) {
                    res = b.create<arith::MinimumFOp>(innerLoc, args[0], args[1]);
                }
                else if (auto maxf = dyn_cast<arith::MinNumFOp>(op)) {
                    res = b.create<arith::MinNumFOp>(innerLoc, args[0], args[1]);
                }
                else if (auto mini = dyn_cast<arith::MaxNumFOp>(op)) {
                    res = b.create<arith::MaxNumFOp>(innerLoc, args[0], args[1]);
                }
                else if (auto maxi = dyn_cast<arith::MaximumFOp>(op)) {
                    res = b.create<arith::MaximumFOp>(innerLoc, args[0], args[1]);
                }
                else if (auto minsi = dyn_cast<arith::MinSIOp>(op)) {
                    res = b.create<arith::MinSIOp>(innerLoc, args[0], args[1]);
                }
                else if (auto maxsi = dyn_cast<arith::MaxSIOp>(op)) {
                    res = b.create<arith::MaxSIOp>(innerLoc, args[0], args[1]);
                }

                b.create<linalg::YieldOp>(innerLoc, ValueRange{res});
            }
        );

        rewriter.replaceOp(srcOp, genericOp->getResults());

        genericOp->setAttr("arithCmpOptimized", BoolAttr::get(rewriter.getContext(), true));

        return success();
    }
};

void populateOptimizeArithElementwiseBinaryOpPatterns(
    MLIRContext *context, RewritePatternSet &patterns
) {
    patterns.add<ArithElementwiseBinaryOpPattern>(context);
}

} // namespace mlir::syna::torq