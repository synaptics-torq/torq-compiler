// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/IR/Types.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ComputeConstants.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include <cstdint>

// Set to false to disable weights inflation and reordering
#define TORQHL_WEIGHT_REORDER true

#define DEBUG_TYPE "torq-patterns"

namespace mlir::syna::torq {

namespace {

llvm::cl::opt<bool> clEnableTosaIdentity(
    "torq-enable-tosa-identity", llvm::cl::desc("Enable identity pattern for tosa ops"),
    llvm::cl::init(false)
);

static bool typeIsInt8(Type type) {
    Type elementType = getElementTypeOrSelf(type);
    return elementType.isSignedInteger() && elementType.getIntOrFloatBitWidth() == 8;
}

static bool checkOpIsInt8(Operation *op) {
    if (!op->getNumResults()) {
        return false;
    }
    Type elementType = getElementTypeOrSelf(op->getResult(0).getType());
    if (typeIsInt8(elementType)) {
        return false;
    }

    return true;
}

LogicalResult getFromConst(Value value, ElementsAttr &attr) {
    if (auto tosaConstOp = value.getDefiningOp<tosa::ConstOp>()) {
        attr = tosaConstOp.getValues();
        return success();
    }
    // Fall back to arith.constant
    if (auto arithConstOp = value.getDefiningOp<arith::ConstantOp>()) {
        attr = mlir::cast<mlir::ElementsAttr>(arithConstOp.getValue());
        return success();
    }
    return failure();
}

struct ArgMaxOpConversion : public OpConversionPattern<tosa::ArgMaxOp> {

    ArgMaxOpConversion(MLIRContext *context) : OpConversionPattern(context) {}
    LogicalResult matchAndRewrite(
        tosa::ArgMaxOp srcOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {
        if (!checkOpIsInt8(srcOp)) {
            return failure();
        }

        Location loc = srcOp.getLoc();
        Value inputVal = adaptor.getInput();
        int64_t axisVal = srcOp.getAxis();

        // Only do transpose trick if argmax on axis=2 on an [N, C, H] layout
        // so we can perform argmax on axis=1 in [N, H, C].
        bool transposeInput = (axisVal == 2);
        if (transposeInput) {
            inputVal = convertNCHtoNHC(inputVal, loc, rewriter);
            axisVal = 1;
        }

        auto axisAttr = rewriter.getI32IntegerAttr(axisVal);
        const std::vector<APInt> bias = {APInt(32, 0)};
        const std::vector<APInt> scale = {APInt(32, 1)};

        rewriter.replaceOpWithNewOp<syna::torq_hl::ArgMaxOp>(
            srcOp, srcOp.getOutput().getType(), createInitTensor(srcOp, rewriter), axisAttr,
            createIConst(rewriter, srcOp, interleave(bias, scale)), inputVal
        );

        return success();
    }
};

struct IdentityOpConversion : public OpConversionPattern<tosa::IdentityOp> {

    IdentityOpConversion(MLIRContext *context) : OpConversionPattern(context) {}
    LogicalResult matchAndRewrite(
        tosa::IdentityOp srcOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {
        rewriter.replaceOpWithNewOp<torq_hl::IdentityOp>(
            srcOp, srcOp.getOutput().getType(), createInitTensor(srcOp, rewriter),
            adaptor.getInput1()
        );
        return success();
    }
};

struct ScatterOpConversion : public OpConversionPattern<tosa::ScatterOp> {
    ScatterOpConversion(MLIRContext *context) : OpConversionPattern(context) {}

    LogicalResult matchAndRewrite(
        tosa::ScatterOp srcOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {
        auto valuesIn = convertNHCtoNCH(adaptor.getValuesIn(), srcOp.getLoc(), rewriter);
        auto indices = srcOp.getIndices();

        // TODO Values are currently loaded from const,
        // but it may come from input.
        // Can check when we have proper model

        ElementsAttr valuesAttr;

        if (failed(getFromConst(indices, valuesAttr))) {
            return rewriter.notifyMatchFailure(srcOp, "Cannot scatter non-constant indices");
        }

        auto values = valuesAttr.getValues<int32_t>();

        std::vector<int16_t> valuesRef(values.begin(), values.end());
        auto indices_value = createI16Const(rewriter, srcOp, valuesRef);

        auto input = convertNHCtoNCH(adaptor.getInput(), srcOp.getLoc(), rewriter);

        auto outType = convertTypeNHCtoNCH(srcOp.getValuesIn().getType());

        auto output = torq_hl::ScatterOp::create(
            rewriter, srcOp.getLoc(), outType, valuesIn, indices_value, input,
            createI32Const(rewriter, srcOp, std::vector<int32_t>{0, 1})
        );

        auto transposeOp = convertNCHtoNHC(output.getOutput(), srcOp.getLoc(), rewriter);
        rewriter.replaceOp(srcOp, transposeOp);
        return success();
    }
};

} // namespace

void populateTOSAToTorqHLPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    if (clEnableTosaIdentity) {
        patterns.insert<IdentityOpConversion>(context);
    }
    patterns.insert<ArgMaxOpConversion>(context);
    patterns.insert<ScatterOpConversion>(context);
}

} // namespace mlir::syna::torq
