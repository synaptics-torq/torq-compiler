// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

#include <optional>

#define DEBUG_TYPE "linalg-torq-clamp-pattern"

namespace mlir::syna::torq {

namespace {

enum class ClampVariant { IntOrdered, Float };

struct ClampInfo {
    ClampVariant variant;
    int32_t minInt = 0;
    int32_t maxInt = 0;
    float minFloat = 0.0f;
    float maxFloat = 0.0f;
    // Which input carries the data (0 unless the clamp is a bound-first form
    // like ins(%bound, %data)).
    int dataInputIdx = 0;
};

// Integer clamp: minsi/maxsi with in-body constants (the float forms are
// handled by matchFloatClampRange).
static bool matchIntClamp(linalg::GenericOp srcOp, ClampInfo &info) {
    std::string failReason;
    if (!isTorqClampOp(srcOp, info.minInt, info.maxInt, info.minFloat, info.maxFloat, failReason))
        return false;
    auto inputType = dyn_cast<RankedTensorType>(srcOp.getInputs()[0].getType());
    if (!inputType || !inputType.getElementType().isInteger())
        return false;
    info.variant = ClampVariant::IntOrdered;
    return true;
}

// Float clamp: the canonical matcher (matchFloatClampRange) recognizes every
// float clamp form -- one- and two-sided, cmpf/select or min/max, ordered or
// unordered, bounds inlined / outer / constant tensor inputs, data in any
// input position, optional trailing width conversion -- and lowers to the same
// single torq_hl.act clamp as the integer form.
static bool matchClamp(linalg::GenericOp srcOp, ClampInfo &info) {
    if (auto range = matchFloatClampRange(srcOp)) {
        info.variant = ClampVariant::Float;
        info.minFloat = range->lo;
        info.maxFloat = range->hi;
        info.dataInputIdx = range->dataInputIdx;
        return true;
    }
    return matchIntClamp(srcOp, info);
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

        Value input = srcOp.getInputs()[clampInfo.dataInputIdx];
        Value output = srcOp.getResultTensors()[0];

        // The clamp's own result, captured before buildFusionPlanAndRebindOutput()
        // rebinds `output` to the downstream fused result. The fallback (non-fusible)
        // path must mark from this original result, otherwise the backward walk starts
        // at the rebound downstream value and sweeps the whole producer cone.
        Value markOrigin = srcOp.getResultTensors()[0];

        // Post-clamp requantise chains only exist on integer paths (int8 conv2d
        // producing i32 followed by tosa.apply_scale + trunc). Float clamps never
        // have a fusible tail, so skip the IR walk entirely for f32/bf16.
        auto outputElemTy = mlir::cast<RankedTensorType>(output.getType()).getElementType();
        FailureOr<FusionPlan> fusionPlanOr = buildFusionPlanAndRebindOutput(output);
        bool hasFusionPlan =
            !outputElemTy.isF32() && !outputElemTy.isBF16() && succeeded(fusionPlanOr);

        if (!hasFusionPlan || !fusionPlanOr->isFusable()) {
            if (_markFuseGroups) {
                if (!markFuseGroupBackward(
                        markOrigin, {input}, rewriter,
                        srcOp->getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
                    ))
                    return failure();
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

        auto finalType = cast<RankedTensorType>(output.getType());

        if (_markFuseGroups) {
            if (!markFuseGroupBackward(
                    output, {input}, rewriter, srcOp->getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
                ))
                return failure();
            return success();
        }

        ScaleClampInfo scInfo = getDefaultScaleClampInfo(finalType, srcOp);
        std::vector<int32_t> a{0};
        std::vector<int32_t> b{1};
        Value biasScale = createI32Const(rewriter, srcOp, interleave(a, b));

        FailureOr<Value> biasV = computeRescaleInfo(*fusionPlanOr, biasScale, scInfo);
        if (failed(biasV)) {
            return rewriter.notifyMatchFailure(srcOp, "Failed to extract rescale info");
        }

        for (auto &op : llvm::reverse(fusionPlanOr->opsToFuse)) {
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
        auto actOp = torq_hl::ActOp::create(
            rewriter, srcOp.getLoc(), srcResultTy, createInitTensor(srcOp, rewriter, srcResultTy),
            "clamp", 0, scInfo.zp, clampInfo.minInt, clampInfo.maxInt,
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
