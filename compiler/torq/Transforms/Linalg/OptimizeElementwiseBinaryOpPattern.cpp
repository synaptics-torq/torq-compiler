// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Utils/ShapeUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-optimize-elementwise-binary-op-pattern"

namespace mlir::syna::torq {

// Canonicalize tensor-output constant linalg.generic into linalg.fill so it can
// be lowered by downstream fill conversion patterns.
class GenericConstantToFillPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {
        if (std::optional<Value> fillValue = isaFillOpInterface(srcOp)) {
            // Always use the detected fill value, regardless of pattern
            rewriter.replaceOpWithNewOp<linalg::FillOp>(srcOp, *fillValue, srcOp.getDpsInits()[0]);
            return success();
        }
        return failure();
    }
};

// elementwise binary ops with 2 inputs

class RewriteIfScalarConstantPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;
    // Rewrites a multi dimensional scalar value passed to a linalg genericOp as single dim scalar
    // Ex: tensor<1x1x1xbf16> -> tensor<1xbf16>
    // While doing this any linalg iteration map has to be changed. Since the input is multi
    // dimensional with 1x1x1 or similar shape the indexing map will have affine dim expressions
    // like (d0, d1, d2) -> (d0, d1, d2) or (d0, d1, d2) -> (d0, d1, 0) etc. After the rewrite the
    // indexing map has to be changed to (d0, d1, d2) -> (0)
    LogicalResult rewriteIfScalarConstant(
        linalg::GenericOp gOp, OpOperand &operand, PatternRewriter &rewriter
    ) const {
        auto denseAttr = returnDenseElementAttr(operand.get());
        if (!denseAttr || denseAttr.getType().getRank() == 1 || denseAttr.getNumElements() != 1) {
            return failure();
        }

        auto iteratorTypes = gOp.getIteratorTypesArray();
        if (!llvm::all_of(iteratorTypes, [](utils::IteratorType iter) {
                return iter == utils::IteratorType::parallel;
            })) {
            return failure();
        }
        ShapedType type = cast<ShapedType>(operand.get().getType());

        auto elementType = type.getElementType();
        auto newType = RankedTensorType::get({1}, elementType);
        auto loc = operand.get().getLoc();
        auto newConst =
            arith::ConstantOp::create(rewriter, loc, newType, denseAttr.reshape(newType));

        auto opIdx = operand.getOperandNumber();
        auto itrMap = gOp.getIndexingMapsArray();
        SmallVector<AffineMap> newIndexingMaps;
        newIndexingMaps.reserve(itrMap.size());

        for (auto [i, map] : llvm::enumerate(itrMap)) {
            if (i == opIdx) {
                auto newMap = AffineMap::get(
                    map.getNumDims(), map.getNumSymbols(), {rewriter.getAffineConstantExpr(0)},
                    rewriter.getContext()
                );
                newIndexingMaps.push_back(newMap);
            }
            else {
                newIndexingMaps.push_back(map);
            }
        }

        rewriter.modifyOpInPlace(gOp, [&]() {
            gOp.setOperand(opIdx, newConst);
            gOp.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(newIndexingMaps));
        });

        return success();
    }

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {
        // If the generic op matches a broadcast then don't do anything because the broadcast
        // pattern will handle the scalar constant and we don't want to interfere with that
        if (linalg::isaBroadcastOpInterface(srcOp)) {
            return failure();
        }

        // Only handle elementwise binary ops for now, we can add more ops later if needed
        bool changed = false;
        for (auto operand : srcOp.getDpsInputOperands()) {
            if (succeeded(rewriteIfScalarConstant(srcOp, *operand, rewriter))) {
                changed = true;
            }
        }
        if (changed) {
            return success();
        }
        return failure();
    }
};

class BroadcastElementwiseBinaryOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    bool isScalarFromRecursiveRescale(const Value &v) const {
        Value input = v;
        ScaleInfo scaleInfo;

        // if input is from tensor.expandshape, assign input to expandop's input
        if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(input.getDefiningOp())) {
            input = expandOp.getSrc();
        }

        bool foldedRescale = false;
        while (foldBackwardRescale(input, scaleInfo)) {
            foldedRescale = true;
        }

        // If foldBackwardRescale traced through rescale(s) to a scalar constant, done.
        if (foldedRescale) {
            if (auto constOp = dyn_cast_if_present<arith::ConstantOp>(input.getDefiningOp())) {
                auto tensorType = dyn_cast<RankedTensorType>(input.getType());
                if (tensorType && tensorType.getNumElements() == 1)
                    return true;
            }
        }

        linalg::GenericOp rescaleOp = input.getDefiningOp<linalg::GenericOp>();

        if (!rescaleOp) {
            LLVM_DEBUG({ llvm::errs() << "Value input definingOp is not linalg.generic op\n"; });
            return false;
        }

        auto yieldOp = dyn_cast<linalg::YieldOp>(rescaleOp.getBody()->getTerminator());
        if (!yieldOp) {
            LLVM_DEBUG({ llvm::errs() << "There is no yield in linalg.generic body\n"; });
            return false;
        }

        auto yieldValues = yieldOp.getValues();
        if (yieldValues.size() != 1) {
            LLVM_DEBUG({ llvm::errs() << "Linalg.yield operand is not 1 \n"; });
            return false;
        }

        tosa::ApplyScaleOp applyScaleOp = yieldValues[0].getDefiningOp<tosa::ApplyScaleOp>();
        if (!applyScaleOp) {
            LLVM_DEBUG({ llvm::errs() << "apply scale op does not exist\n"; });
            return false;
        }

        Value value = applyScaleOp.getValue();
        if (!value) {
            LLVM_DEBUG({ llvm::errs() << "applyScaleOp cannot get value\n"; });
            return false;
        }

        if (auto constOp = dyn_cast_if_present<arith::ConstantOp>(value.getDefiningOp())) {
            int32_t data = 0;
            if (getIntegerConstantValue(constOp, &data))
                return true;
        }

        LLVM_DEBUG({ llvm::errs() << "apply_scale value is not from a scalar constant\n"; });
        return false;
    }

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {

        if (srcOp->getAttrOfType<BoolAttr>("broadcasted")) {
            return failure();
        }

        if (srcOp.getInputs().empty() || srcOp.getInputs().size() > 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "elementwise binary op pattern expects generic op with 2 (or 1) inputs\n"
            );
        }

        if (srcOp.getNumResults() != 1) {
            return rewriter.notifyMatchFailure(
                srcOp, "elementwise binary op pattern expects generic op with 1 output\n"
            );
        }

        auto resultType = dyn_cast<RankedTensorType>(srcOp.getResultTypes().front());
        if (!resultType) {
            return rewriter.notifyMatchFailure(
                srcOp, "elementwise binary op pattern expects ranked tensor result type\n"
            );
        }
        auto resultElemType = resultType.getElementType();
        if (resultElemType.isF64()) {
            return rewriter.notifyMatchFailure(
                srcOp, "elementwise binary op pattern does not support f64\n"
            );
        }

        auto srcOpSize = srcOp.getNumDpsInputs();
        if (srcOpSize == 1) {
            return failure();
        }

        SmallVector<Value> inputs(srcOp.getInputs().begin(), srcOp.getInputs().end());
        Value &input1 = inputs[0];
        Value &input2 = inputs[1];

        auto input1Type = dyn_cast<RankedTensorType>(input1.getType());
        if (!input1Type) {
            return rewriter.notifyMatchFailure(
                srcOp, "elementwise binary op pattern expects ranked tensor input1 type\n"
            );
        }
        auto input2Type = dyn_cast<RankedTensorType>(input2.getType());
        if (!input2Type) {
            return rewriter.notifyMatchFailure(
                srcOp, "elementwise binary op pattern expects ranked tensor input2 type\n"
            );
        }

        if (input1Type == resultType && input2Type == resultType) {
            return failure();
        }

        Operation *eleOp = getElementwiseBinaryOp(srcOp, true);

        arith::ShRSIOp shrsiOp1;
        bool isRRSOp = isRoundingRightShiftOp(srcOp, shrsiOp1);

        if (!eleOp && !isRRSOp) {
            return rewriter.notifyMatchFailure(
                srcOp, "generic op is not expected elementwise binary op\n"
            );
        }

        // TODO: add more elementwise binary ops
        if (eleOp && !isa<arith::AddIOp>(eleOp) && !isa<arith::AddFOp>(eleOp) &&
            !isa<arith::SubIOp>(eleOp) && !isa<arith::SubFOp>(eleOp) &&
            !isa<arith::MulIOp>(eleOp) && !isa<arith::MulFOp>(eleOp) &&
            !isa<arith::DivFOp>(eleOp) && !isa<arith::CmpIOp>(eleOp)) {
            return rewriter.notifyMatchFailure(
                srcOp, "elementwise binary op pattern only supports add/sub/mul/cmp ..\n"
            );
        }

        auto rank1 = input1Type.getRank();
        auto rank2 = input2Type.getRank();

        if (rank1 == 0 || rank2 == 0) {
            return rewriter.notifyMatchFailure(
                srcOp, "one of input or both input rank is 0, no need broadcast\n"
            );
        }
        auto isScalar = [&](Value input) {
            if (isa_and_nonnull<arith::ConstantOp>(input.getDefiningOp()) &&
                returnDenseElementAttr(input).getNumElements() == 1) {
                return true;
            }
            return false;
        };

        if (isScalar(input1) || isScalar(input2)) {
            return rewriter.notifyMatchFailure(
                srcOp, "one of input or both input is scalar, no need broadcast\n"
            );
        }

        // TODO: add more recursive scalar input processing for elementwise binary ops
        // right now we only handle add/sub with recurive scalar input processing
        if (eleOp && (isa<arith::AddIOp>(eleOp) || isa<arith::SubIOp>(eleOp))) {
            if (isScalarFromRecursiveRescale(input1) || isScalarFromRecursiveRescale(input2)) {
                return rewriter.notifyMatchFailure(
                    srcOp, "one of input or both input is recurive scalar, no need broadcast\n"
                );
            }
        }

        if (failed(broadcastInputs(srcOp, inputs, rewriter))) {
            return rewriter.notifyMatchFailure(
                srcOp, "failed to broadcast inputs for elementwise binary op\n"
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

void populateOptimizeElementwiseBinaryOpPatterns(
    MLIRContext *context, RewritePatternSet &patterns
) {
    patterns.add<RewriteIfScalarConstantPattern>(context);
    patterns.add<GenericConstantToFillPattern>(context);
    patterns.add<BroadcastElementwiseBinaryOpPattern>(context);
}

} // namespace mlir::syna::torq
