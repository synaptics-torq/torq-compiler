// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "OpPatternOptions.h"
#include "torq/Conversions/LinalgToTorqHL/MatchingFunctions.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-torq-clamp-pattern"

namespace mlir::syna::torq {

namespace {

enum class ClampVariant { Ordered, NaiveOrdered, Unordered, ReLU };

struct ClampInfo {
    ClampVariant variant;
    int32_t minInt = 0;
    int32_t maxInt = 0;
    float minFloat = 0.0f;
    float maxFloat = 0.0f;
};

static float toFloat(arith::ConstantOp cst) {
    return mlir::cast<FloatAttr>(cst.getValue()).getValue().convertToFloat();
}

// Shared precondition for the two cmpf/select matchers: single f32/bf16 input,
// single output, exactly 5 ops in the body.
static bool isFloatClampCandidate(linalg::GenericOp op) {
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
        return false;
    auto inputType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
    if (!inputType)
        return false;
    auto elemTy = inputType.getElementType();
    return (elemTy.isF32() || elemTy.isBF16()) &&
           op.getRegion().front().getOperations().size() == 5;
}

// Ordered clamp: arith.minimumf/maximumf (float) or arith.minsi/maxsi (int).
//
//   %0 = linalg.generic ... ins(%in : tensor<Nxf32>) outs(%out : tensor<Nxf32>) {
//   ^bb0(%arg0: f32, %arg1: f32):
//     %min = arith.minimumf %arg0, %cst_max : f32
//     %max = arith.maximumf %min, %cst_min : f32
//     linalg.yield %max : f32
//   }
static bool matchOrderedClamp(linalg::GenericOp srcOp, ClampInfo &info) {
    std::string failReason;
    if (!isTorqClampOp(srcOp, info.minInt, info.maxInt, info.minFloat, info.maxFloat, failReason))
        return false;
    info.variant = ClampVariant::Ordered;
    return true;
}

// Naive ordered clamp from BfloatTanhPattern.
//
//   %0 = linalg.generic ... ins(%in : tensor<Nxbf16>) outs(%out : tensor<Nxbf16>) {
//   ^bb0(%arg0: bf16, %arg1: bf16):
//     %cmp0 = arith.cmpf olt, %cst_max, %arg0 : bf16
//     %sel0 = arith.select %cmp0, %cst_max, %arg0 : bf16
//     %cmp1 = arith.cmpf ogt, %cst_min, %sel0 : bf16
//     %sel1 = arith.select %cmp1, %cst_min, %sel0 : bf16
//     linalg.yield %sel1 : bf16
//   }
static bool matchNaiveOrderedClamp(linalg::GenericOp srcOp, ClampInfo &info) {
    if (!isFloatClampCandidate(srcOp))
        return false;

    auto &block = srcOp.getRegion().front();

    auto cmpfOlt = dyn_cast<arith::CmpFOp>(block.front());
    if (!cmpfOlt || cmpfOlt.getPredicate() != arith::CmpFPredicate::OLT)
        return false;
    auto cstMax = cmpfOlt.getLhs().getDefiningOp<arith::ConstantOp>();
    auto inArg = dyn_cast<BlockArgument>(cmpfOlt.getRhs());
    if (!cstMax || !inArg || inArg.getArgNumber() != 0)
        return false;

    auto select1 = dyn_cast<arith::SelectOp>(cmpfOlt->getNextNode());
    if (!select1 || select1.getCondition() != cmpfOlt.getResult())
        return false;
    if (!select1.getTrueValue().getDefiningOp<arith::ConstantOp>())
        return false;
    auto inArg2 = dyn_cast<BlockArgument>(select1.getFalseValue());
    if (!inArg2 || inArg2.getArgNumber() != 0)
        return false;

    auto cmpfOgt = dyn_cast<arith::CmpFOp>(select1->getNextNode());
    if (!cmpfOgt || cmpfOgt.getPredicate() != arith::CmpFPredicate::OGT)
        return false;
    auto cstMin = cmpfOgt.getLhs().getDefiningOp<arith::ConstantOp>();
    if (!cstMin || cmpfOgt.getRhs() != select1.getResult())
        return false;

    auto select2 = dyn_cast<arith::SelectOp>(cmpfOgt->getNextNode());
    if (!select2 || select2.getCondition() != cmpfOgt.getResult())
        return false;
    if (!select2.getTrueValue().getDefiningOp<arith::ConstantOp>())
        return false;
    if (select2.getFalseValue() != select1.getResult())
        return false;

    auto yieldOp = dyn_cast<linalg::YieldOp>(select2->getNextNode());
    if (!yieldOp || yieldOp.getValues().size() != 1 ||
        yieldOp.getValues()[0] != select2.getResult())
        return false;

    info.variant = ClampVariant::NaiveOrdered;
    info.minFloat = toFloat(cstMin);
    info.maxFloat = toFloat(cstMax);
    return true;
}

// Unordered clamp from IREE ConvertElementwiseToLinalgPass (ONNX Clip/ReLU6).
//
//   %0 = linalg.generic ... ins(%in : tensor<Nxbf16>) outs(%out : tensor<Nxbf16>) {
//   ^bb0(%arg0: bf16, %arg1: bf16):
//     %cmp0 = arith.cmpf ult, %arg0, %cst_min : bf16
//     %sel0 = arith.select %cmp0, %cst_min, %arg0 : bf16
//     %cmp1 = arith.cmpf ugt, %sel0, %cst_max : bf16
//     %sel1 = arith.select %cmp1, %cst_max, %sel0 : bf16
//     linalg.yield %sel1 : bf16
//   }
static bool matchUnorderedClamp(linalg::GenericOp srcOp, ClampInfo &info) {
    if (!isFloatClampCandidate(srcOp))
        return false;

    auto &block = srcOp.getRegion().front();

    auto cmpfUlt = dyn_cast<arith::CmpFOp>(block.front());
    if (!cmpfUlt || cmpfUlt.getPredicate() != arith::CmpFPredicate::ULT)
        return false;
    auto cstMin = cmpfUlt.getRhs().getDefiningOp<arith::ConstantOp>();
    auto inArg = dyn_cast<BlockArgument>(cmpfUlt.getLhs());
    if (!cstMin || !inArg || inArg.getArgNumber() != 0)
        return false;

    auto select1 = dyn_cast<arith::SelectOp>(cmpfUlt->getNextNode());
    if (!select1 || select1.getCondition() != cmpfUlt.getResult())
        return false;
    if (!select1.getTrueValue().getDefiningOp<arith::ConstantOp>())
        return false;
    auto inArg2 = dyn_cast<BlockArgument>(select1.getFalseValue());
    if (!inArg2 || inArg2.getArgNumber() != 0)
        return false;

    auto cmpfUgt = dyn_cast<arith::CmpFOp>(select1->getNextNode());
    if (!cmpfUgt || cmpfUgt.getPredicate() != arith::CmpFPredicate::UGT)
        return false;
    auto cstMax = cmpfUgt.getRhs().getDefiningOp<arith::ConstantOp>();
    if (!cstMax || cmpfUgt.getLhs() != select1.getResult())
        return false;

    auto select2 = dyn_cast<arith::SelectOp>(cmpfUgt->getNextNode());
    if (!select2 || select2.getCondition() != cmpfUgt.getResult())
        return false;
    if (!select2.getTrueValue().getDefiningOp<arith::ConstantOp>())
        return false;
    if (select2.getFalseValue() != select1.getResult())
        return false;

    auto yieldOp = dyn_cast<linalg::YieldOp>(select2->getNextNode());
    if (!yieldOp || yieldOp.getValues().size() != 1 ||
        yieldOp.getValues()[0] != select2.getResult())
        return false;

    info.variant = ClampVariant::Unordered;
    info.minFloat = toFloat(cstMin);
    info.maxFloat = toFloat(cstMax);
    return true;
}

// ReLU: a one-sided lower clamp expressed as a single cmpf/select pair.
//
//   %0 = linalg.generic ... ins(%in : tensor<Nxbf16>) outs(%out : tensor<Nxbf16>) {
//   ^bb0(%arg0: bf16, %arg1: bf16):
//     %cmp = arith.cmpf ugt, %arg0, %cst_min : bf16
//     %sel = arith.select %cmp, %arg0, %cst_min : bf16
//     linalg.yield %sel : bf16
//   }
//
// This is equivalent to clamp(x, cst_min, +inf), i.e. ReLU when cst_min == 0.
static bool matchReLU(linalg::GenericOp srcOp, ClampInfo &info) {
    if (srcOp.getInputs().size() != 1 || srcOp.getOutputs().size() != 1)
        return false;
    auto inputType = dyn_cast<RankedTensorType>(srcOp.getInputs()[0].getType());
    if (!inputType)
        return false;
    auto elemTy = inputType.getElementType();
    if (!elemTy.isF32() && !elemTy.isBF16())
        return false;
    // Body must have exactly 3 ops: cmpf, select, yield.
    if (srcOp.getRegion().front().getOperations().size() != 3)
        return false;

    auto &block = srcOp.getRegion().front();

    // First op: arith.cmpf ugt, %arg0, %cst
    auto cmpOp = dyn_cast<arith::CmpFOp>(block.front());
    if (!cmpOp || cmpOp.getPredicate() != arith::CmpFPredicate::UGT)
        return false;
    auto inArg = dyn_cast<BlockArgument>(cmpOp.getLhs());
    auto cstMin = cmpOp.getRhs().getDefiningOp<arith::ConstantOp>();
    if (!inArg || inArg.getArgNumber() != 0 || !cstMin)
        return false;

    // Second op: arith.select %cmp, %arg0, %cst
    auto selOp = dyn_cast<arith::SelectOp>(cmpOp->getNextNode());
    if (!selOp || selOp.getCondition() != cmpOp.getResult())
        return false;
    auto selTrueArg = dyn_cast<BlockArgument>(selOp.getTrueValue());
    if (!selTrueArg || selTrueArg.getArgNumber() != 0)
        return false;
    if (!selOp.getFalseValue().getDefiningOp<arith::ConstantOp>())
        return false;

    // Third op: linalg.yield %sel
    auto yieldOp = dyn_cast<linalg::YieldOp>(selOp->getNextNode());
    if (!yieldOp || yieldOp.getValues().size() != 1 || yieldOp.getValues()[0] != selOp.getResult())
        return false;

    info.variant = ClampVariant::ReLU;
    info.minFloat = toFloat(cstMin);
    info.maxFloat = std::numeric_limits<float>::max();
    return true;
}

static bool matchClamp(linalg::GenericOp srcOp, ClampInfo &info) {
    return matchOrderedClamp(srcOp, info) || matchNaiveOrderedClamp(srcOp, info) ||
           matchUnorderedClamp(srcOp, info) || matchReLU(srcOp, info);
}

struct ClampOpConversion : public OpRewritePattern<linalg::GenericOp> {
  private:
    const bool _markFuseGroups;

  public:
    using OpRewritePattern::OpRewritePattern;
    ClampOpConversion(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(srcOp))
            return rewriter.notifyMatchFailure(srcOp, "Already marked in a fuse group");

        ClampInfo clampInfo;
        if (!matchClamp(srcOp, clampInfo))
            return rewriter.notifyMatchFailure(srcOp, "Not a recognised clamp variant");

        LLVM_DEBUG({
            llvm::dbgs() << "ClampOpConversion: variant=" << static_cast<int>(clampInfo.variant)
                         << " min=" << clampInfo.minFloat << " max=" << clampInfo.maxFloat << " : ";
            srcOp.print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
            llvm::dbgs() << "\n";
        });

        Value input = srcOp.getInputs()[0];
        Value output = srcOp.getResultTensors()[0];

        // Post-clamp requantise chains only exist on integer paths (int8 conv2d
        // producing i32 followed by tosa.apply_scale + trunc). Float clamps never
        // have a fusible tail, so skip the IR walk entirely for f32/bf16.
        auto outputElemTy = mlir::cast<RankedTensorType>(output.getType()).getElementType();
        FusionPlan fusionPlan;
        bool hasFusionPlan =
            !outputElemTy.isF32() && !outputElemTy.isBF16() && createFusionPlan(output, fusionPlan);

        if (!hasFusionPlan || fusionPlan.neededOps.empty()) {
            if (_markFuseGroups) {
                markFuseGroupBackward(
                    output, {input}, rewriter, srcOp->getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
                );
                return success();
            }
            auto resultTy = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
            rewriter.replaceOpWithNewOp<torq_hl::ActOp>(
                srcOp, resultTy, createInitTensor(srcOp, rewriter, resultTy), "clamp", 0, 0,
                clampInfo.minInt, clampInfo.maxInt,
                APFloat(llvm::APFloat::IEEEsingle(), std::to_string(clampInfo.minFloat)),
                APFloat(llvm::APFloat::IEEEsingle(), std::to_string(clampInfo.maxFloat)), input,
                /*weights=*/mlir::Value()
            );
            return success();
        }

        output = fusionPlan.neededOps.back()->getResult(0);
        auto finalType = cast<RankedTensorType>(output.getType());

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {input}, rewriter, srcOp->getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        ScaleClampInfo scInfo = getDefaultScaleClampInfo(finalType, srcOp);
        Value biasScale = createI32Const(rewriter, srcOp, interleave({0}, {1}));
        if (!computeRescaleInfo(fusionPlan, /*isElementWiseOp=*/true, biasScale, scInfo))
            return rewriter.notifyMatchFailure(srcOp, "Failed to extract rescale info");

        for (auto &op : llvm::reverse(fusionPlan.opsToFuse)) {
            if (op->use_empty()) {
                LLVM_DEBUG({
                    llvm::dbgs() << "ClampOpConversion erasing: ";
                    op->print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
                    llvm::dbgs() << "\n";
                });
                rewriter.eraseOp(op);
            }
        }

        rewriter.setInsertionPoint(output.getDefiningOp());
        auto srcResultTy = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
        auto actOp = rewriter.create<torq_hl::ActOp>(
            srcOp.getLoc(), srcResultTy, createInitTensor(srcOp, rewriter, srcResultTy), "clamp", 0,
            scInfo.zp, clampInfo.minInt, clampInfo.maxInt,
            APFloat(llvm::APFloat::IEEEsingle(), std::to_string(clampInfo.minFloat)),
            APFloat(llvm::APFloat::IEEEsingle(), std::to_string(clampInfo.maxFloat)), input,
            /*weights=*/mlir::Value()
        );
        rewriter.replaceOp(output.getDefiningOp(), actOp.getResult(0));

        LLVM_DEBUG(llvm::dbgs() << "ClampOpConversion success (with fusion)\n");
        return success();
    }
};

} // namespace

void populateLinalgToTorqHLClampPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<ClampOpConversion>(context, markFuseGroups);
}

} // namespace mlir::syna::torq
