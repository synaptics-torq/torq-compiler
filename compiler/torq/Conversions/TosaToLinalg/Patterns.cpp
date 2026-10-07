// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/IR/Types.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <numeric>
#include <type_traits>

#define DEBUG_TYPE "tosa-to-linalg"

namespace mlir::syna::torq {

namespace {

struct ResizeNearestNeighborOpConversion : public OpConversionPattern<tosa::ResizeOp> {
    ResizeNearestNeighborOpConversion(MLIRContext *context) : OpConversionPattern(context) {}

    LogicalResult matchAndRewrite(
        tosa::ResizeOp srcOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {

        SmallVector<int64_t> scale;
        if (!tosa::getConstShapeValues(srcOp.getScale().getDefiningOp(), scale)) {
            return failure();
        }
        if (scale.size() != 4) {
            return rewriter.notifyMatchFailure(srcOp, "expected four resize scale values");
        }

        int64_t scaleHNumerator = scale[0];
        int64_t scaleHDenominator = scale[1];
        int64_t scaleWNumerator = scale[2];
        int64_t scaleWDenominator = scale[3];
        if (scaleHDenominator <= 0 || scaleWDenominator <= 0) {
            return rewriter.notifyMatchFailure(srcOp, "expected positive scale denominators");
        }
        if (scaleHNumerator < scaleHDenominator || scaleWNumerator < scaleWDenominator) {
            return rewriter.notifyMatchFailure(srcOp, "expected height and width upscale ratios");
        }
        if (scaleHNumerator % scaleHDenominator != 0 || scaleWNumerator % scaleWDenominator != 0) {
            return rewriter.notifyMatchFailure(
                srcOp, "expected positive integer height and width upscale factors"
            );
        }

        int64_t scaleH = scaleHNumerator / scaleHDenominator;
        int64_t scaleW = scaleWNumerator / scaleWDenominator;
        auto mode = adaptor.getMode();
        if (mode != tosa::ResizeMode::NEAREST_NEIGHBOR) {
            return rewriter.notifyMatchFailure(
                srcOp, "expected NEAREST_NEIGHBOR mode with integer upscale factors"
            );
        }

        // Transpose input NHWC -> NCHW, run the resize in NCHW, then transpose back.
        auto resultTypeNCHW = convertTypeNHWCtoNCHW(srcOp.getResult().getType());
        AffineExpr n, c, h, w;
        bindDims(rewriter.getContext(), n, c, h, w);
        // NCHW maps: input[n, c, h/scaleH, w/scaleW] -> output[n, c, h, w]
        auto inputMap = AffineMap::get(
            4, 0, {n, c, h.floorDiv(scaleH), w.floorDiv(scaleW)}, rewriter.getContext()
        );
        auto outputMap = AffineMap::get(4, 0, {n, c, h, w}, rewriter.getContext());
        SmallVector<AffineMap, 4> indexingMaps = {inputMap, outputMap};
        SmallVector<mlir::utils::IteratorType, 4> iteratorTypes(
            4, mlir::utils::IteratorType::parallel
        );
        auto input = convertNHWCtoNCHW(adaptor.getInput(), srcOp.getLoc(), rewriter);
        auto output = linalg::GenericOp::create(
            rewriter, srcOp.getLoc(), resultTypeNCHW, input,
            createInitTensor(srcOp.getLoc(), resultTypeNCHW, rewriter), indexingMaps, iteratorTypes,
            [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
                linalg::YieldOp::create(nestedBuilder, nestedLoc, args[0]);
            }
        );
        auto outputTransposed = convertNCHWtoNHWC(output.getResult(0), srcOp.getLoc(), rewriter);
        rewriter.replaceOp(srcOp, outputTransposed);

        return success();
    }
};

} // namespace

void populateTOSAToLinalgPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.insert<ResizeNearestNeighborOpConversion>(context);
}

} // namespace mlir::syna::torq
