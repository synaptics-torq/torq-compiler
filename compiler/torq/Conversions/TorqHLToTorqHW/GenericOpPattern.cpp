// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"

#include "torq/Dialect/TorqHL/GenericOp.h"
#include "torq/Utils/CodeSizeUtils.h"
#include "torq/Utils/InvocationUtils.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-lower-torqhl"

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

class GenericOpPattern : public OpConversionPattern<torq_hl::GenericOp> {
  public:
    using OpConversionPattern<torq_hl::GenericOp>::OpConversionPattern;

    int getElementSize(Value value) const {
        auto memRefType = cast<MemRefType>(value.getType());
        return memRefType.getElementTypeBitWidth() / 8;
    }

    std::optional<MemNdlDimsData> createNdl(
        torq_hl::GenericOp op, Value value, AffineMap map, PatternRewriter &rewriter,
        int vectorSize = 1
    ) const {

        auto ctx = value.getContext();

        if (!value) {
            return std::nullopt;
        }

        auto memRefType = dyn_cast<MemRefType>(value.getType());

        if (!memRefType) {
            return std::nullopt;
        }

        auto memRefStrides = getEncodedStridesElements(memRefType);
        auto accessExpr = getAffineConstantExpr(0, ctx);

        for (int i = 0; i < memRefType.getRank(); i++) {
            accessExpr = accessExpr +
                         getAffineDimExpr(i, ctx) * getAffineConstantExpr(memRefStrides[i], ctx);
        }

        auto dAccessMap = AffineMap::get(memRefType.getRank(), 0, accessExpr);

        auto dLoop = dAccessMap.compose(map);

        MemNdlDimsData ndl;

        ndl.push_back(
            {DimType::L, MemDimTag::B, memRefType.getElementTypeBitWidth() / 8 * vectorSize, 1}
        );

        SmallVector<int64_t> ndlStrides;

        // TODO: handle offset
        if (failed(getFlattenedAffineExpr(dLoop.getResult(0), dLoop.getNumDims(), 0, &ndlStrides)
            )) {
            return std::nullopt;
        }

        auto loopCount = op.getStaticLoopRanges();

        for (int i = 0; i < loopCount.size(); i++) {
            ndl.push_back({DimType::H, MemDimTag::O, loopCount[i], ndlStrides[i]});
        }

        return ndl;
    }

    LogicalResult matchAndRewrite(
        torq_hl::GenericOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {

        if (!op.getAluConfig() || !op.getActConfig()) {
            return failure();
        }

        auto biasMap = op.getBiasMap().value();
        auto scaleMap = op.getScaleMap().value();

        // we only support reading scale and bias from the same tensor with specific layout
        if (op.getBias() && op.getScale()) {
            if (op.getBias() != op.getScale()) {
                return failure();
            }

            if (scaleMap.getResult(scaleMap.getNumResults() - 1) !=
                getAffineConstantExpr(0, op.getContext())) {
                return failure();
            }

            if (biasMap.getResult(biasMap.getNumResults() - 1) !=
                getAffineConstantExpr(1, op.getContext())) {
                return failure();
            }

            auto biasType = dyn_cast<ShapedType>(op.getBias().getType());

            if (!biasType) {
                return failure();
            }

            if (biasType.getShape()[biasType.getRank() - 1] != 2) {
                return failure();
            }

            // TODO: we should check also the layout
        }

        auto aluConfig = op.getAluConfig().value();
        auto actConfig = op.getActConfig().value();

        auto ctx = rewriter.getContext();

        ALUOp0Mode aluOp0Mode;

        if (aluConfig.getOp0Mode() == torq_hl::ALUOp0Mode::MUL) {
            aluOp0Mode = ALUOp0Mode::MUL;
        }
        else if (aluConfig.getOp0Mode() == torq_hl::ALUOp0Mode::DBYP) {
            aluOp0Mode = ALUOp0Mode::DBYP;
        }
        else {
            return failure();
        }

        ALUOp1Mode aluOp1Mode;

        if (aluConfig.getOp1Mode() == torq_hl::ALUOp1Mode::ACC) {
            aluOp1Mode = ALUOp1Mode::ACC;
        }
        else {
            return failure();
        }

        if (!adaptor.getD() || !adaptor.getBias() || !adaptor.getScale()) {
            return failure();
        }

        if (aluOp0Mode != ALUOp0Mode::DBYP && !adaptor.getW()) {
            return failure();
        }

        auto slice_cfg_attr = torq_hw::SliceCFGAttr::get(
            ctx, {aluOp0Mode, aluOp0Mode, aluOp0Mode, aluOp0Mode},
            {aluOp1Mode, aluOp1Mode, aluOp1Mode, aluOp1Mode},
            0,                                  // alu_d_unsigned
            0,                                  // alu_w_unsigned
            ACTMode::ACT,                       // act_mode
            {0, 0, 0, 0},                       // act left shift
            actConfig.getShiftFactorDiv4() * 4, // shift_factor
            actConfig.getOutputMin(),           // output_min
            actConfig.getOutputMax(),           // output_max
            actConfig.getOutputZeroPoint(),     // output_zp
            false,                              // no_p_clear
            false,                              // no_p_output
            1, 1, 1, 1,                         // kernel lrtb
            0, 0, 0, 0,                         // pad lrtb
            0,                                  // pad_value
            1                                   // stride
        );

        // for the moment we support only reductions in the last dimensions
        bool foundReduceDim = false;
        for (auto dimType : op.getIteratorTypesArray()) {
            if (dimType == utils::IteratorType::reduction) {
                foundReduceDim = true;
            }
            else if (dimType == utils::IteratorType::parallel) {
                if (foundReduceDim) {
                    return failure();
                }
            }
        }

        auto bSize = getElementSize(adaptor.getBias());
        auto pSize = getElementSize(adaptor.getP());

        // we don't use the padding so we don't need to set the ref ndl
        MemNdlDimsData ref;

        auto maybeDedr = createNdl(op, adaptor.getD(), op.getDMap().value(), rewriter);

        if (!maybeDedr) {
            return failure();
        }

        auto maybeDeqw = createNdl(op, adaptor.getQ(), op.getQMap().value(), rewriter);

        if (!maybeDeqw) {
            return failure();
        }

        // here we use the scale map because we checked the bias ans scale maps and value
        // allow to do it above
        auto maybeDebr = createNdl(op, adaptor.getBias(), scaleMap, rewriter, 2);

        if (!maybeDebr) {
            return failure();
        }

        // copy whatever was loaded by cedr into IRAM row #0
        RegNdlDimsData cedw = {
            {DimType::L, RegDimTag::B, 1, 1}, // copy one byte at time
            {DimType::L, RegDimTag::D, 64, 1} // copy 64 blocks of 1 bytes
        };

        // read back whatever was loaded into IRAM row #0
        RegNdlDimsData cedr = {
            {DimType::L, RegDimTag::B, 1, 1},  // copy one byte at time
            {DimType::L, RegDimTag::D, 64, 1}, // copy 64 blocks of 1 bytes
            {DimType::L, RegDimTag::S, 1, 1}   // this dimension does nothing but is mandatory
        };

        MemNdlDimsData dewr;
        RegNdlDimsData ceww;
        RegNdlDimsData cewr;

        if (aluOp0Mode != ALUOp0Mode::DBYP) {
            auto wSize = getElementSize(adaptor.getW());

            auto maybeDewr = createNdl(op, adaptor.getW(), op.getWMap().value(), rewriter);

            if (!maybeDewr) {
                return failure();
            }

            dewr = *maybeDewr;

            ceww = {{DimType::L, RegDimTag::B, wSize, 1}, {DimType::L, RegDimTag::D, 1, wSize}};

            cewr = {{DimType::L, RegDimTag::B, wSize, 1}, {DimType::L, RegDimTag::D, 1, wSize}};
        }

        RegNdlDimsData cepr = {
            {DimType::L, RegDimTag::B, pSize, 1}, {DimType::L, RegDimTag::D, 1, pSize}
        };

        RegNdlDimsData acbw = {
            {DimType::L, RegDimTag::B, bSize, 1}, {DimType::L, RegDimTag::D, 1, bSize}
        };

        RegNdlDimsData acbr = {
            {DimType::L, RegDimTag::B, bSize * 2, 1}, {DimType::L, RegDimTag::D, 1, bSize}
        };

        RegNdlDimsData acpr = {
            {DimType::L, RegDimTag::B, pSize, 1},
            {DimType::L, RegDimTag::D, HwInfo::act_width, HwInfo::pram_dsize
            },                                // required by the hardware
            {DimType::L, RegDimTag::S, 1, 1}, // required by the hardware
            {DimType::H, RegDimTag::T, 0, 1}  // this forces the acpr to keep running continuously
        };

        Ndls ndls;
        ndls.add(NdlType::REF, ref);
        ndls.add(NdlType::DEDR, *maybeDedr);
        ndls.add(NdlType::DEWR, dewr);
        ndls.add(NdlType::DEBR, *maybeDebr);
        ndls.add(NdlType::DEQW, *maybeDeqw);
        ndls.add(NdlType::CEDW, cedw);
        ndls.add(NdlType::CEDR, cedr);
        ndls.add(NdlType::CEWW, ceww);
        ndls.add(NdlType::CEWR, cewr);
        ndls.add(NdlType::CEPR, cepr);
        ndls.add(NdlType::ACBW, acbw);
        ndls.add(NdlType::ACBR, acbr);
        ndls.add(NdlType::ACPR, acpr);

        auto pVal = op.getP();

        rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
            op, "generic", adaptor.getD(), adaptor.getW(), adaptor.getBias(), adaptor.getQ(),
            slice_cfg_attr, ndls
        );

        // delete the p buffer if nobody uses it
        bool eraseP = true;
        for (auto pUser : pVal.getUsers()) {
            if (!isa<torq_hl::LoadOp>(pUser)) {
                eraseP = false;
                break;
            }
        }

        if (eraseP) {
            rewriter.eraseOp(pVal.getDefiningOp());
        }

        return success();
    }
};

void populateGenericOpPatterns(MLIRContext *ctx, RewritePatternSet &patterns) {
    patterns.add<GenericOpPattern>(ctx);
}

} // namespace mlir::syna::torq
