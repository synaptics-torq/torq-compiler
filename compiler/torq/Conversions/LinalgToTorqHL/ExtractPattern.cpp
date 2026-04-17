// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "OpPatternOptions.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ComputeConstants.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "linalg-torq-extract-pattern"

namespace mlir::syna::torq {

// TODO: only support int8 for now in order to use generic tiling for table op
class ExtractOpPattern : public OpRewritePattern<linalg::GenericOp> {

    LogicalResult validateGatherValuesRank(
        linalg::GenericOp srcOp, Value values, PatternRewriter &rewriter,
        RankedTensorType &valuesType
    ) const {
        valuesType = dyn_cast<RankedTensorType>(values.getType());
        if (!valuesType) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected gathered values to be a ranked tensor"
            );
        }

        int64_t valuesRank = valuesType.getRank();
        if (valuesRank < 1 || valuesRank > 3) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected gathered values rank to be 1, 2, or 3"
            );
        }

        return success();
    }

    LogicalResult demoteGatherIndicesToI32(
        linalg::GenericOp srcOp, Value indices, PatternRewriter &rewriter, Value &demotedIndices
    ) const {
        demotedIndices = indices;

        auto indicesType = dyn_cast<RankedTensorType>(indices.getType());
        if (!indicesType) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected gather indices to be ranked tensor"
            );
        }

        auto indicesElementType = dyn_cast<IntegerType>(indicesType.getElementType());
        if (!indicesElementType) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected gather indices to be integer tensor"
            );
        }

        if (indicesElementType.getWidth() <= 32) {
            return success();
        }

        auto i32IndicesType = RankedTensorType::get(
            indicesType.getShape(), rewriter.getI32Type(), indicesType.getEncoding()
        );

        auto maybeConstAttr = computeConstAttr(indices, /*recursive=*/true);
        if (failed(maybeConstAttr)) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected gather indices i64 tensor to be constant or constant-foldable"
            );
        }

        auto indicesConstAttr = dyn_cast<DenseIntElementsAttr>(*maybeConstAttr);
        if (!indicesConstAttr) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected gather indices constant to be DenseIntElementsAttr"
            );
        }

        SmallVector<APInt> i32IndicesValues;
        i32IndicesValues.reserve(indicesConstAttr.getNumElements());
        for (APInt idx : indicesConstAttr.getValues<APInt>()) {
            if (!idx.isSignedIntN(32)) {
                return rewriter.notifyMatchFailure(
                    srcOp, "Expected gather indices constant values to fit in i32"
                );
            }
            i32IndicesValues.push_back(idx.sextOrTrunc(32));
        }

        auto i32IndicesAttr = DenseIntElementsAttr::get(i32IndicesType, i32IndicesValues);
        demotedIndices =
            arith::ConstantOp::create(rewriter, srcOp.getLoc(), i32IndicesType, i32IndicesAttr)
                .getResult();
        return success();
    }

    // GatherOp treats indices as linear offsets into the flattened values buffer.
    // For supported sliced gathers (rank > 1 within the rank-1/2/3 cases),
    // rescale constant outer-dim indices so they point at the start of each
    // contiguous slice in that flattened layout.
    LogicalResult scaleGatherIndicesForSlice(
        linalg::GenericOp srcOp, Value values, Value indices, PatternRewriter &rewriter,
        Value &scaledIndices
    ) const {
        scaledIndices = indices;

        auto valuesType = dyn_cast<RankedTensorType>(values.getType());
        if (!valuesType || valuesType.getRank() <= 1) {
            return success();
        }

        int64_t sliceStride = 1;
        for (int64_t dim : valuesType.getShape().drop_front()) {
            if (dim == ShapedType::kDynamic) {
                return rewriter.notifyMatchFailure(
                    srcOp, "Expected gather source slice stride to be static"
                );
            }
            sliceStride *= dim;
        }
        if (sliceStride == 1) {
            return success();
        }

        auto maybeConstAttr = computeConstAttr(indices, /*recursive=*/true);
        if (failed(maybeConstAttr)) {
            return success();
        }

        auto indicesAttr = dyn_cast<DenseIntElementsAttr>(*maybeConstAttr);
        if (!indicesAttr) {
            return success();
        }

        auto indicesType = dyn_cast<RankedTensorType>(indices.getType());
        auto elementType =
            indicesType ? dyn_cast<IntegerType>(indicesType.getElementType()) : nullptr;
        if (!indicesType || !elementType) {
            return success();
        }

        SmallVector<APInt> scaledValues;
        scaledValues.reserve(indicesAttr.getNumElements());
        APInt scale(64, sliceStride, /*isSigned=*/true);
        for (APInt idx : indicesAttr.getValues<APInt>()) {
            APInt scaled = idx.sextOrTrunc(64) * scale;
            if (!scaled.isSignedIntN(elementType.getWidth())) {
                return rewriter.notifyMatchFailure(
                    srcOp, "Expected scaled gather indices to fit in the element type"
                );
            }
            scaledValues.push_back(scaled.sextOrTrunc(elementType.getWidth()));
        }

        auto scaledAttr = DenseIntElementsAttr::get(indicesType, scaledValues);
        scaledIndices = arith::ConstantOp::create(rewriter, srcOp.getLoc(), indicesType, scaledAttr)
                            .getResult();
        return success();
    }

  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {

        if (srcOp.getInputs().size() != 1 || srcOp.getResults().size() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "Expects 1 input and 1 output");
        }

        Value input = srcOp.getInputs()[0];

        auto yieldOp = dyn_cast<linalg::YieldOp>(srcOp.getBody()->getTerminator());
        if (!yieldOp) {
            return rewriter.notifyMatchFailure(srcOp, "Expected a linalg.yield terminator");
        }

        auto tensorExtractOp = yieldOp.getValues()[0].getDefiningOp<tensor::ExtractOp>();
        if (!tensorExtractOp) {
            return rewriter.notifyMatchFailure(srcOp, "Expected a tensor.extract op");
        }

        if (failed(demoteGatherIndicesToI32(srcOp, input, rewriter, input))) {
            return rewriter.notifyMatchFailure(
                srcOp, "Failed to demote tensor.extract indices to int32"
            );
        }
        auto inputType = dyn_cast<RankedTensorType>(input.getType());
        auto inputElementType = inputType.getElementType();

        auto &block = srcOp.getRegion().front();
        auto &firstOp = block.front();

        auto indices = tensorExtractOp.getIndices();
        bool isTableOp = false;
        if (!indices.empty()) {
            Value idx = indices[0];

            auto addOp = dyn_cast<arith::AddIOp>(idx.getDefiningOp());
            if (addOp) {
                isTableOp = true;
                if (!inputElementType.isInteger(8)) {
                    return rewriter.notifyMatchFailure(srcOp, "Only i8 TableOp supported");
                }

                // Check the table tensor element type. If the LUT is i16
                // (from swish table fusion), defer to I16ExtractTableOpPattern.
                auto tableType = dyn_cast<RankedTensorType>(tensorExtractOp.getTensor().getType());
                if (tableType && !tableType.getElementType().isInteger(8)) {
                    return rewriter.notifyMatchFailure(
                        srcOp, "Table tensor is not i8; deferring to I16ExtractTableOpPattern"
                    );
                }

                auto indexCastOp = dyn_cast<arith::IndexCastOp>(firstOp);
                if (!indexCastOp) {
                    return rewriter.notifyMatchFailure(
                        srcOp, "Expected firstOp is a arith.indexCast op"
                    );
                }
                // one operands of addOp is indexCastOp result
                Value indexCastResult = indexCastOp.getResult();
                if (indexCastResult != addOp.getLhs() && indexCastResult != addOp.getRhs()) {
                    return rewriter.notifyMatchFailure(
                        srcOp, "we expect one addOp operand from indexCastOp"
                    );
                }
            }
        }

        Value cst = tensorExtractOp.getTensor();
        if (!cst) {
            return rewriter.notifyMatchFailure(srcOp, "Expected a arith.addi op");
        }

        RankedTensorType cstType;
        if (failed(validateGatherValuesRank(srcOp, cst, rewriter, cstType))) {
            return failure();
        }

        SmallVector<int32_t> convertedValues;
        auto maybeCstData = computeArithConst(cst);
        if (failed(maybeCstData)) {
            // tensor is input
            auto resultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

            if (cstType.getRank() == 3) {
                // Rank-3 gathers come from NHC values and need an NHC->NCH
                // transpose before lowering to GatherOp.
                auto dataPerm = Permutation::nhc2nch();
                auto outType = transposeType(resultType, dataPerm.reverse());
                auto transposedValue = transposeValue(cst, dataPerm, srcOp.getLoc(), rewriter);
                auto out = syna::torq_hl::GatherOp::create(
                    rewriter, srcOp.getLoc(), outType, createInitTensor(srcOp, rewriter, outType),
                    transposedValue, input
                );
                auto resultTranspose =
                    transposeValue(out.getResult(0), dataPerm.reverse(), srcOp.getLoc(), rewriter);
                rewriter.replaceOp(srcOp, resultTranspose);
            }
            else {
                // Rank-1 gathers are direct scalar lookups. Rank-2 gathers use
                // GatherOp's flat linear-offset path and need their constant
                // outer-dim indices rescaled by the trailing slice stride.
                if (failed(scaleGatherIndicesForSlice(srcOp, cst, input, rewriter, input))) {
                    return rewriter.notifyMatchFailure(
                        srcOp, "Failed to scale constant gather indices for sliced gathers"
                    );
                }
                rewriter.replaceOpWithNewOp<syna::torq_hl::GatherOp>(
                    srcOp, resultType, createInitTensor(srcOp, rewriter, resultType), cst, input
                );
            }
        }

        else if (!clTableAsGather && isTableOp) {
            auto values = attrValuesAsVec<int8_t>(*maybeCstData);
            for (size_t i = 0; i < 256; i++) {
                int32_t shiftedValue = static_cast<int32_t>(values[i]) << 8;
                convertedValues.push_back(shiftedValue);
            }

            DenseI32ArrayAttr intArrayAttr =
                DenseI32ArrayAttr::get(rewriter.getContext(), convertedValues);

            const std::vector<APInt> bias = {APInt(32, -128, /*isSigned=*/true)};
            const std::vector<APInt> scale = {APInt(32, 128, /*isSigned=*/true)};

            auto resultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
            rewriter.replaceOpWithNewOp<syna::torq_hl::TableOp>(
                srcOp, resultType, createInitTensor(srcOp, rewriter, resultType),
                createIConst(rewriter, srcOp, interleave(bias, scale)), input, intArrayAttr
            );
        }
        else if (isTableOp) { // If a table converted Gather then the const values are input(table)
                              // and extract tensor is indices
            std::vector<APInt> tableValues;
            auto values = attrValuesAsVec<APInt>(*maybeCstData);
            tableValues.insert(tableValues.end(), values.begin(), values.end());
            int tableSize = tableValues.size();
            std::vector<APInt> modifiedTable;
            modifiedTable.reserve(tableSize);
            modifiedTable.insert(modifiedTable.end(), tableValues.begin() + 128, tableValues.end());
            modifiedTable.insert(
                modifiedTable.end(), tableValues.begin(), tableValues.begin() + 128
            );
            tableValues = modifiedTable;

            auto outType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
            rewriter.replaceOpWithNewOp<syna::torq_hl::GatherOp>(
                srcOp, outType, createInitTensor(srcOp, rewriter, outType),
                createIConst(rewriter, srcOp, tableValues), input
            );
        }
        else { // If not a table converted Gather and the const values are
               // input(table)
               // and extract tensor is indices

            auto outType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
            if (failed(scaleGatherIndicesForSlice(srcOp, cst, input, rewriter, input))) {
                return rewriter.notifyMatchFailure(
                    srcOp, "Failed to scale constant gather indices for sliced gathers"
                );
            }

            rewriter.replaceOpWithNewOp<syna::torq_hl::GatherOp>(
                srcOp, outType, createInitTensor(srcOp, rewriter, outType), *maybeCstData, input
            );
        }

        return success();
    }
};

// Table lookup with i16 LUT values.
// Supports two input variants:
//   (a) i16 input: trunci(i16→i8) + index_cast + addi(128) + extract from i16 LUT
//   (b) i8 input:  index_cast + addi(128) + extract from i16 LUT  (swish table fusion)
//
// Both produce i16 output from a 256-entry i16 lookup table.
// The +128 offset converts signed i8 range [-128,127] to unsigned index [0,255].
class I16ExtractTableOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {

        // Must have exactly 1 input and 1 output.
        if (srcOp.getInputs().size() != 1 || srcOp.getResults().size() != 1)
            return rewriter.notifyMatchFailure(srcOp, "i16Table: expects 1 input and 1 output");

        Value input = srcOp.getInputs()[0];
        auto inputType = dyn_cast<RankedTensorType>(input.getType());
        if (!inputType)
            return rewriter.notifyMatchFailure(srcOp, "i16Table: input must be a ranked tensor");

        auto inputElemType = inputType.getElementType();
        bool inputIsI8 = inputElemType.isInteger(8);
        bool inputIsI16 = inputElemType.isInteger(16);
        if (!inputIsI8 && !inputIsI16)
            return rewriter.notifyMatchFailure(srcOp, "i16Table: input must be i8 or i16");

        // Check yield.
        auto yieldOp = dyn_cast<linalg::YieldOp>(srcOp.getBody()->getTerminator());
        if (!yieldOp)
            return rewriter.notifyMatchFailure(srcOp, "i16Table: expected linalg.yield");

        // The yielded value must come from tensor.extract.
        auto extractOp = yieldOp.getValues()[0].getDefiningOp<tensor::ExtractOp>();
        if (!extractOp)
            return rewriter.notifyMatchFailure(srcOp, "i16Table: expected tensor.extract");

        // Walk the body depending on input type.
        auto &block = srcOp.getRegion().front();
        auto *firstOp = &block.front();

        arith::IndexCastOp indexCastOp;
        if (inputIsI16) {
            // Variant (a): trunci -> index_cast -> addi(128) -> extract
            auto trunciOp = dyn_cast<arith::TruncIOp>(firstOp);
            if (!trunciOp)
                return rewriter.notifyMatchFailure(
                    srcOp, "i16Table: i16 input expects arith.trunci first"
                );
            indexCastOp = dyn_cast_or_null<arith::IndexCastOp>(trunciOp->getNextNode());
            if (!indexCastOp)
                return rewriter.notifyMatchFailure(
                    srcOp, "i16Table: expected arith.index_cast after trunci"
                );
        }
        else {
            // Variant (b): index_cast -> addi(128) -> extract  (i8 input, no trunci)
            indexCastOp = dyn_cast<arith::IndexCastOp>(firstOp);
            if (!indexCastOp)
                return rewriter.notifyMatchFailure(
                    srcOp, "i16Table: i8 input expects arith.index_cast first"
                );
        }

        // The index fed into tensor.extract must be addi(%index_cast, 128).
        auto indices = extractOp.getIndices();
        if (indices.empty())
            return rewriter.notifyMatchFailure(srcOp, "i16Table: extract has no indices");

        auto addOp = dyn_cast<arith::AddIOp>(indices[0].getDefiningOp());
        if (!addOp)
            return rewriter.notifyMatchFailure(srcOp, "i16Table: index must come from arith.addi");

        Value indexCastResult = indexCastOp.getResult();
        if (indexCastResult != addOp.getLhs() && indexCastResult != addOp.getRhs())
            return rewriter.notifyMatchFailure(srcOp, "i16Table: addi operand not from index_cast");

        // Verify the table tensor is tensor<256xi16>.
        Value tableTensor = extractOp.getTensor();
        auto tableType = dyn_cast<RankedTensorType>(tableTensor.getType());
        if (!tableType || tableType.getRank() != 1 || tableType.getDimSize(0) != 256 ||
            !tableType.getElementType().isInteger(16))
            return rewriter.notifyMatchFailure(srcOp, "i16Table: table must be tensor<256xi16>");

        // Fold the table constant.
        auto maybeCstData = computeArithConst(tableTensor);
        if (failed(maybeCstData))
            return rewriter.notifyMatchFailure(srcOp, "i16Table: table tensor is not constant");

        // Pack 256 i16 values into 256 i32 entries: (slope << 16) | (base & 0xFFFF).
        // Slope is set to 0 for now.
        auto values = attrValuesAsVec<int16_t>(*maybeCstData);
        SmallVector<int32_t> convertedValues;
        convertedValues.reserve(256);
        for (size_t i = 0; i < 256; i++) {
            int16_t tb = values[i];         // base value
            int32_t packed = (tb & 0xFFFF); // slope=0, base only
            convertedValues.push_back(packed);
        }

        DenseI32ArrayAttr intArrayAttr =
            DenseI32ArrayAttr::get(rewriter.getContext(), convertedValues);

        const std::vector<APInt> bias = {APInt(32, -128, /*isSigned=*/true)};
        const std::vector<APInt> scale = {APInt(32, 128, /*isSigned=*/true)};

        auto resultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
        rewriter.replaceOpWithNewOp<syna::torq_hl::TableOp>(
            srcOp, resultType, createInitTensor(srcOp, rewriter, resultType),
            createIConst(rewriter, srcOp, interleave(bias, scale)), input, intArrayAttr
        );

        return success();
    }
};

void populateLinalgToTorqHLExtractPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<ExtractOpPattern>(context);
    patterns.insert<I16ExtractTableOpPattern>(context);
}

} // namespace mlir::syna::torq
