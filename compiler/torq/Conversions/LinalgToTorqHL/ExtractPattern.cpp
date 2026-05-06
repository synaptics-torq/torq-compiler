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
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/ExecutorAssignment.h"

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

// Builds packed_table: tensor<N×i32> from an i8 or i16 source tensor.
// For i8 (signExtend=true):  packed[i] = sext(i8) << 8   (slope=0, base-only encoding)
// For i16 (signExtend=false): packed[i] = zext(i16)        (base-only: 0<<16 | (tb & 0xFFFF))
static Value buildPackedTable(OpBuilder &builder, Location loc, Value src, bool signExtend) {
    auto srcType = cast<RankedTensorType>(src.getType());
    auto i32Type = builder.getI32Type();
    auto outType = RankedTensorType::get(srcType.getShape(), i32Type);
    Value empty = tensor::EmptyOp::create(builder, loc, outType.getShape(), i32Type).getResult();
    SmallVector<AffineMap> maps{2, AffineMap::getMultiDimIdentityMap(1, builder.getContext())};
    SmallVector<utils::IteratorType> iterTypes{utils::IteratorType::parallel};
    return linalg::GenericOp::create(
               builder, loc, outType, src, empty, maps, iterTypes, "", "",
               [signExtend, i32Type](OpBuilder &b, Location loc, ValueRange args) {
                   Value ext = signExtend ? (Value)arith::ExtSIOp::create(b, loc, i32Type, args[0])
                                          : (Value)arith::ExtUIOp::create(b, loc, i32Type, args[0]);
                   if (signExtend) {
                       Value shift8 = arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(8));
                       ext = arith::ShLIOp::create(b, loc, ext, shift8);
                   }
                   linalg::YieldOp::create(b, loc, ext);
               }
    ).getResult(0);
}

// Builds the interleaved (bias, scale) constant for TableOp's scale_bias operand.
// Both i8 and i16 table variants use the same fixed encoding: bias=-128, scale=128.
static Value buildTableScaleBias(linalg::GenericOp srcOp, PatternRewriter &rewriter) {
    const std::vector<APInt> bias = {APInt(32, -128, /*isSigned=*/true)};
    const std::vector<APInt> scale = {APInt(32, 128, /*isSigned=*/true)};
    return createIConst(rewriter, srcOp, interleave(bias, scale));
}

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

    // Detect whether `tensorExtractOp` is a table-lookup (LUT) op by checking
    // that the extract index is produced by an addi whose one operand is the
    // result of the index_cast at the top of the block (the +128 bias pattern).
    // Sets `isTableOp` to true when the pattern is recognised and validates:
    //   - input element type is i8,
    //   - the LUT tensor element type is i8 (non-i8 defers to I16ExtractTableOpPattern),
    //   - the first body op is an index_cast whose result feeds the addi.
    // Returns failure() if the pattern is partially recognised but invalid.
    // Returns success() with isTableOp=false when the pattern is absent.
    LogicalResult detectTableOp(
        linalg::GenericOp srcOp, tensor::ExtractOp tensorExtractOp, Type inputElementType,
        Operation &firstOp, PatternRewriter &rewriter, bool &isTableOp
    ) const {
        isTableOp = false;

        auto indices = tensorExtractOp.getIndices();
        if (indices.empty())
            return success();

        auto addOp = indices[0].getDefiningOp<arith::AddIOp>();
        if (!addOp)
            return success();

        isTableOp = true;

        if (!inputElementType.isInteger(8))
            return rewriter.notifyMatchFailure(srcOp, "Only i8 TableOp supported");

        // If the LUT is i16 (from swish table fusion), defer to I16ExtractTableOpPattern.
        auto tableType = dyn_cast<RankedTensorType>(tensorExtractOp.getTensor().getType());
        if (tableType && !tableType.getElementType().isInteger(8))
            return rewriter.notifyMatchFailure(
                srcOp, "Table tensor is not i8; deferring to I16ExtractTableOpPattern"
            );

        auto indexCastOp = dyn_cast<arith::IndexCastOp>(firstOp);
        if (!indexCastOp)
            return rewriter.notifyMatchFailure(srcOp, "Expected firstOp is a arith.indexCast op");

        // One operand of addOp must be the indexCastOp result.
        Value indexCastResult = indexCastOp.getResult();
        if (indexCastResult != addOp.getLhs() && indexCastResult != addOp.getRhs())
            return rewriter.notifyMatchFailure(
                srcOp, "we expect one addOp operand from indexCastOp"
            );

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

        // Demote i64 → i32 via a linalg.generic { arith.trunci }.
        // FIXME: arith.trunci silently drops the high bits. If gather indices can ever
        // exceed [INT32_MIN, INT32_MAX] in practice, we need a compile time or runtime check to
        // ensure safety.
        Location loc = srcOp.getLoc();
        Value emptyDemoted =
            tensor::EmptyOp::create(rewriter, loc, i32IndicesType.getShape(), rewriter.getI32Type())
                .getResult();
        int64_t rank = indicesType.getRank();
        SmallVector<AffineMap> maps{
            2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext())
        };
        SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
        demotedIndices =
            linalg::GenericOp::create(
                rewriter, loc, i32IndicesType, indices, emptyDemoted, maps, iterTypes, "", "",
                [](OpBuilder &b, Location loc, ValueRange args) {
                    Value trunc = arith::TruncIOp::create(b, loc, b.getI32Type(), args[0]);
                    linalg::YieldOp::create(b, loc, trunc);
                }
            ).getResult(0);
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

        auto trailingDims = valuesType.getShape().drop_front();
        if (llvm::any_of(trailingDims, ShapedType::isDynamic))
            return rewriter.notifyMatchFailure(
                srcOp, "Expected gather source slice stride to be static"
            );
        int64_t sliceStride = ShapedType::getNumElements(trailingDims);
        if (sliceStride == 1) {
            return success();
        }

        auto indicesType = dyn_cast<RankedTensorType>(indices.getType());
        auto elementType =
            indicesType ? dyn_cast<IntegerType>(indicesType.getElementType()) : nullptr;
        if (!indicesType || !elementType) {
            return success();
        }

        // Emit scaledIndices[i] = indices[i] * sliceStride as IR ops so both
        // constant and dynamic index tensors are handled uniformly.
        Location loc = srcOp.getLoc();
        Value emptyScaled =
            tensor::EmptyOp::create(rewriter, loc, indicesType.getShape(), elementType).getResult();
        int64_t rank = indicesType.getRank();
        SmallVector<AffineMap> maps{
            2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext())
        };
        SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
        scaledIndices =
            linalg::GenericOp::create(
                rewriter, loc, indicesType, indices, emptyScaled, maps, iterTypes, "", "",
                [&, sliceStride](OpBuilder &b, Location loc, ValueRange args) {
                    Value stride = arith::ConstantOp::create(
                        b, loc, b.getIntegerAttr(args[0].getType(), sliceStride)
                    );
                    Value scaled = arith::MulIOp::create(b, loc, args[0], stride);
                    linalg::YieldOp::create(b, loc, scaled);
                }
            ).getResult(0);
        setCompileTimeConstAttr(scaledIndices.getDefiningOp());
        return success();
    }

    // Lower a dynamic (non-constant) values tensor to GatherOp.
    // Rank-3 values are in NHC layout and need a transpose to NCH around the gather.
    // Rank-1/2 values use GatherOp's flat-offset path; rank-2 outer-dim indices
    // are rescaled by the trailing slice stride before lowering.
    LogicalResult rewriteAsDynamicGather(
        linalg::GenericOp srcOp, Value valuesTensor, RankedTensorType valuesTensorType, Value input,
        PatternRewriter &rewriter
    ) const {
        auto resultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

        if (valuesTensorType.getRank() == 3) {
            // Rank-3 gathers come from NHC values and need an NHC->NCH
            // transpose before lowering to GatherOp.
            auto dataPerm = Permutation::nhc2nch();
            auto outType = transposeType(resultType, dataPerm.reverse());
            auto transposedValue = transposeValue(valuesTensor, dataPerm, srcOp.getLoc(), rewriter);
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
            if (failed(scaleGatherIndicesForSlice(srcOp, valuesTensor, input, rewriter, input)))
                return rewriter.notifyMatchFailure(
                    srcOp, "Failed to scale constant gather indices for sliced gathers"
                );
            rewriter.replaceOpWithNewOp<syna::torq_hl::GatherOp>(
                srcOp, resultType, createInitTensor(srcOp, rewriter, resultType), valuesTensor,
                input
            );
        }
        return success();
    }

    // Lower a constant i8 LUT to TableOp. Each i8 entry is packed as `value << 8`
    // into an i32 word (slope=0, base-only piecewise-linear encoding).
    void rewriteAsTableOp(
        linalg::GenericOp srcOp, Value input, Value cstData, PatternRewriter &rewriter
    ) const {
        Location loc = srcOp.getLoc();
        Value packedTable = buildPackedTable(rewriter, loc, cstData, /*signExtend=*/true);
        setCompileTimeConstAttr(packedTable.getDefiningOp());
        auto resultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
        rewriter.replaceOpWithNewOp<syna::torq_hl::TableOp>(
            srcOp, resultType, createInitTensor(srcOp, rewriter, resultType),
            buildTableScaleBias(srcOp, rewriter), input, packedTable, nullptr
        );
    }

    // Lower a constant i8 LUT to GatherOp when `clTableAsGather` is set.
    // Rotates the table by 128 positions so that signed i8 indices [-128,127]
    // map to unsigned GatherOp indices [0,255].
    void rewriteTableAsGather(
        linalg::GenericOp srcOp, Value input, Value cstData, PatternRewriter &rewriter
    ) const {
        // Rotate by 128: move entries [128,255] to the front, [0,127] to the back,
        // so signed i8 indices [-128,127] map to unsigned [0,255].
        Location loc = srcOp.getLoc();
        auto cstType = cast<RankedTensorType>(cstData.getType());
        Type elemTy = cstType.getElementType();
        constexpr int64_t half = 128;
        auto halfType = RankedTensorType::get({half}, elemTy);
        auto mkIdx = [&](int64_t v) -> OpFoldResult { return rewriter.getIndexAttr(v); };

        Value upperHalf = tensor::ExtractSliceOp::create(
                              rewriter, loc, halfType, cstData, ArrayRef<OpFoldResult>{mkIdx(half)},
                              ArrayRef<OpFoldResult>{mkIdx(half)}, ArrayRef<OpFoldResult>{mkIdx(1)}
        )
                              .getResult();
        Value lowerHalf = tensor::ExtractSliceOp::create(
                              rewriter, loc, halfType, cstData, ArrayRef<OpFoldResult>{mkIdx(0)},
                              ArrayRef<OpFoldResult>{mkIdx(half)}, ArrayRef<OpFoldResult>{mkIdx(1)}
        )
                              .getResult();
        Value rotated =
            tensor::EmptyOp::create(rewriter, loc, cstType.getShape(), elemTy).getResult();
        rotated = tensor::InsertSliceOp::create(
                      rewriter, loc, upperHalf, rotated, ArrayRef<OpFoldResult>{mkIdx(0)},
                      ArrayRef<OpFoldResult>{mkIdx(half)}, ArrayRef<OpFoldResult>{mkIdx(1)}
        )
                      .getResult();
        rotated = tensor::InsertSliceOp::create(
                      rewriter, loc, lowerHalf, rotated, ArrayRef<OpFoldResult>{mkIdx(half)},
                      ArrayRef<OpFoldResult>{mkIdx(half)}, ArrayRef<OpFoldResult>{mkIdx(1)}
        )
                      .getResult();

        setCompileTimeConstAttr(rotated.getDefiningOp());
        auto outType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
        rewriter.replaceOpWithNewOp<syna::torq_hl::GatherOp>(
            srcOp, outType, createInitTensor(srcOp, rewriter, outType), rotated, input
        );
    }

    // Lower a plain constant values tensor to GatherOp.
    // Rank-2 outer-dim indices are rescaled by the trailing slice stride.
    LogicalResult rewriteAsConstantGather(
        linalg::GenericOp srcOp, Value valuesTensor, Value input, PatternRewriter &rewriter
    ) const {
        auto outType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
        if (failed(scaleGatherIndicesForSlice(srcOp, valuesTensor, input, rewriter, input)))
            return rewriter.notifyMatchFailure(
                srcOp, "Failed to scale constant gather indices for sliced gathers"
            );
        rewriter.replaceOpWithNewOp<syna::torq_hl::GatherOp>(
            srcOp, outType, createInitTensor(srcOp, rewriter, outType), valuesTensor, input
        );
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

        auto yieldOp = cast<linalg::YieldOp>(srcOp.getBody()->getTerminator());

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

        bool isTableOp = false;
        if (failed(detectTableOp(
                srcOp, tensorExtractOp, inputElementType, firstOp, rewriter, isTableOp
            )))
            return failure();

        Value valuesTensor = tensorExtractOp.getTensor();

        RankedTensorType valuesTensorType;
        if (failed(validateGatherValuesRank(srcOp, valuesTensor, rewriter, valuesTensorType)))
            return failure();

        auto maybeConst = outlineAndReturnOps(valuesTensor);
        bool valuesAreConstant = succeeded(maybeConst);

        if (valuesAreConstant) {
            setCompileTimeConstAttr(valuesTensor.getDefiningOp());
        }
        if (!valuesAreConstant)
            return rewriteAsDynamicGather(srcOp, valuesTensor, valuesTensorType, input, rewriter);

        if (isTableOp && !clTableAsGather) {
            rewriteAsTableOp(srcOp, input, valuesTensor, rewriter);
            return success();
        }
        if (isTableOp) { // clTableAsGather == true: emit the LUT as a rotated GatherOp
            rewriteTableAsGather(srcOp, input, valuesTensor, rewriter);
            return success();
        }
        return rewriteAsConstantGather(srcOp, valuesTensor, input, rewriter);
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
        auto yieldOp = cast<linalg::YieldOp>(srcOp.getBody()->getTerminator());

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

        auto addOp = indices[0].getDefiningOp<arith::AddIOp>();
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

        // Guard: table must be a compile-time constant so TablePattern can fold it.
        if (failed(outlineAndReturnOps(tableTensor, /*recursive=*/true)))
            return rewriter.notifyMatchFailure(srcOp, "i16Table: table tensor is not constant");

        setCompileTimeConstAttr(tableTensor.getDefiningOp());
        Location loc = srcOp.getLoc();
        Value packedTable = buildPackedTable(rewriter, loc, tableTensor, /*signExtend=*/false);
        auto resultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
        rewriter.replaceOpWithNewOp<syna::torq_hl::TableOp>(
            srcOp, resultType, createInitTensor(srcOp, rewriter, resultType),
            buildTableScaleBias(srcOp, rewriter), input, packedTable, nullptr
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
