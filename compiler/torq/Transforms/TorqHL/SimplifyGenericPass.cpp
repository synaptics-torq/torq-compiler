// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "GenericOpPassesDetail.h"

#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/GenericOp.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include <memory>
#include <tuple>

#define DEBUG_TYPE "torq-simplify-generic"

namespace mlir::syna::torq_hl {

namespace {

class MergePInitialization : public OpRewritePattern<torq_hl::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hl::GenericOp genericOp, PatternRewriter &rewriter) const override {

        if (!genericOp.getAluConfig() || genericOp.getActConfig()) {
            return failure();
        }

        if (genericOp.getAluConfig().value().getOp1Mode() != torq_hl::ALUOp1Mode::ACC) {
            return failure();
        }

        auto pOp = genericOp.getP().getDefiningOp<linalg::GenericOp>();

        if (!pOp)
            return failure();

        if (pOp.getNumDpsInits() != 1 || pOp.getNumDpsInputs() != 1)
            return failure();

        auto pConstVal = pOp.getDpsInputOperand(0)->get().getDefiningOp<arith::ConstantOp>();

        if (!pConstVal)
            return failure();

        auto yieldOp = cast<linalg::YieldOp>(pOp.getBody()->getTerminator());

        auto yieldArg = dyn_cast<BlockArgument>(yieldOp.getValues()[0]);

        if (!yieldArg || yieldArg.getOwner() != pOp.getBody() || yieldArg.getArgNumber() != 0)
            return failure();

        auto pOpOutMap = pOp.getMatchingIndexingMap(pOp.getDpsInitOperand(0));
        auto pOpInMap = pOp.getMatchingIndexingMap(pOp.getDpsInputOperand(0));
        auto genericOpPMap = genericOp.getPMap();

        // A : represents pOp
        // B : represents genericOp
        // C : represent the rewritten genericOp
        //
        // 1. A[AOutMap(dA)] := Ain[AInMap(dA)]
        //
        // 2. B_Q[BQMap(dB)] := B_P[BPMap(dB)]
        // 3. B_P[BPMap(dB)] := Reduce[B_D(DMap(dB))] + B_Pinit[BPMap(dbB)]
        //
        // we can rewrite 1 as follows:
        //
        // 4. A[dA] = A[AOutMap(AOutMapInv(dA))] = Ain[AInMap(AoutMapInv(dA))]
        //
        // inserting 4. in 3. we get:
        //
        // 5. B_P[BPMap(dB)] := Reduce[B_D(DMap(dB))] + Ain[AInMap(AoutMapInv(BPMap(dbB)))]
        //
        // inserting 5. in 2 we get:
        //
        // 6. B_Q[BQMap(dB)] := Reduce[B_D(DMap(dB))] + Ain[AInMap(AoutMapInv(BPMap(dbB)))]
        //
        // we can therefore create a new operation C as with Ain[AInMap(AoutMapInv(BPMap(dbB)))] as
        // bias and use Pinit = 0 while retaining the same output.
        //
        // The access map for the bias is AInMap(AoutMapInv(BPMap(dbB))), if we ensure AoutMap is an
        // identity we then have AInMap(AoutMapInv(BPMap(dbB))) = AInMap(BPMap(dbB))

        if (!pOpOutMap.isIdentity()) {
            return failure();
        }

        auto biasMap = pOpInMap.compose(genericOpPMap);

        Value emptyTensor = rewriter.create<tensor::EmptyOp>(
            genericOp.getLoc(), genericOp.getP().getType(), ValueRange{}
        );

        Value zeroConst =
            rewriter.create<arith::ConstantOp>(genericOp.getLoc(), rewriter.getI32IntegerAttr(0));

        Value zeroTensor =
            rewriter.create<linalg::FillOp>(genericOp.getLoc(), zeroConst, emptyTensor).result();

        GenericOpConfig config = GenericOpConfig::fromOperation(genericOp);

        config.p = GenericOpParam(zeroTensor, config.p.map());
        config.bias = GenericOpParam(pConstVal, biasMap);

        config.actConfig = ActConfigAttr::get(rewriter.getContext(), 0, 0, ACT_MIN, ACT_MAX);

        rewriter.replaceOpWithNewOp<torq_hl::GenericOp>(genericOp, config);

        return success();
    }
};

class MergeBiasScale : public OpRewritePattern<torq_hl::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hl::GenericOp scaleOp, PatternRewriter &rewriter) const override {

        if (scaleOp.getBias() || !scaleOp.getScale() || !scaleOp.getActConfig()) {
            return failure();
        }

        if (scaleOp.getD() || scaleOp.getW() || scaleOp.getAluConfig()) {
            return failure();
        }

        auto biasOp = scaleOp.getP().getDefiningOp<torq_hl::GenericOp>();

        if (!biasOp)
            return failure();

        if (biasOp.getD() || biasOp.getW() || biasOp.getScale() || biasOp.getAluConfig()) {
            return failure();
        }

        if (!biasOp.getBias() || !biasOp.getActConfig()) {
            return failure();
        }

        if (biasOp.getActConfig().value().getOutputMax() != ACT_MAX ||
            biasOp.getActConfig().value().getOutputMin() != ACT_MIN ||
            biasOp.getActConfig().value().getShiftFactorDiv4() != 0 ||
            biasOp.getActConfig().value().getOutputZeroPoint() != 0) {
            return failure();
        }

        if (scaleOp.getPMap() != biasOp.getPMap()) {
            return failure();
        }

        if (scaleOp.getQMap() != biasOp.getQMap()) {
            return failure();
        }

        GenericOpConfig config = GenericOpConfig::fromOperation(scaleOp);

        config.p = GenericOpParam(biasOp.getP(), biasOp.getPMapAttr());
        config.bias = GenericOpParam(biasOp.getBias(), biasOp.getBiasMapAttr());

        rewriter.replaceOpWithNewOp<torq_hl::GenericOp>(scaleOp, config);

        return success();
    }
};

class MergeAluAct : public OpRewritePattern<torq_hl::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hl::GenericOp actOp, PatternRewriter &rewriter) const override {

        if (!actOp.getActConfig()) {
            return failure();
        }

        if (actOp.getAluConfig() || !actOp.getQ()) {
            return failure();
        }

        auto aluOp = actOp.getP().getDefiningOp<torq_hl::GenericOp>();

        // TODO: we could be more general but we need to do some work more here
        // with the maps
        if (!actOp.getPMap().isIdentity() || !actOp.getQMap().value().isIdentity()) {
            return failure();
        }

        Value newBias = actOp.getBias();

        if (aluOp.getActConfig()) {

            if (aluOp.getActConfig().value().getOutputMax() != ACT_MAX ||
                aluOp.getActConfig().value().getOutputMin() != ACT_MIN ||
                aluOp.getActConfig().value().getShiftFactorDiv4() != 0 ||
                aluOp.getActConfig().value().getOutputZeroPoint() != 0) {
                return failure();
            }

            // TODO: here we check the two maps corresponds so that we can do a simple addition
            // below we should handle the general case by inserting the correct linalg.generic
            if (actOp.getBiasMap().value().compose(aluOp.getPMap()) != aluOp.getBiasMap()) {
                return failure();
            }

            Value emptyTensor = rewriter.create<tensor::EmptyOp>(
                actOp.getLoc(), actOp.getBias().getType(), ValueRange{}
            );

            newBias = rewriter
                          .create<linalg::AddOp>(
                              actOp.getLoc(), ValueRange{actOp.getBias(), aluOp.getBias()},
                              ValueRange{emptyTensor}
                          )
                          .getResult(0);
        }

        GenericOpConfig config = GenericOpConfig::fromOperation(aluOp);

        if (actOp.getScale()) {
            auto newScaleMap = actOp.getScaleMap().value().compose(aluOp.getPMap());
            config.scale = GenericOpParam(actOp.getScale(), newScaleMap);
        }

        if (actOp.getBias()) {
            auto newBiasMap = actOp.getBiasMap().value().compose(aluOp.getPMap());
            config.bias = GenericOpParam(newBias, newBiasMap);
        }

        config.actConfig = actOp.getActConfigAttr();
        auto newQMap = actOp.getQMap().value().compose(aluOp.getPMap());
        config.q = GenericOpParam(actOp.getQ(), newQMap);

        rewriter.replaceOpWithNewOp<torq_hl::GenericOp>(actOp, config);

        return success();
    }
};

class SimplifyGenericPass : public SimplifyGenericBase<SimplifyGenericPass> {
  public:
    using SimplifyGenericBase::SimplifyGenericBase;

    void runOnOperation() override {

        auto &context = getContext();

        RewritePatternSet patterns(&context);

        patterns.add<MergePInitialization>(&context);

        // we apply the patter because as we it is written it won't match after
        // we merge ACT and ALU (which may happend with the next patterns)
        // TODO: we need to generalize the above pattern to handle this case
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            getOperation().emitError() << "pass failed";
            return signalPassFailure();
        }

        RewritePatternSet patterns2(&context);

        patterns2.add<MergeBiasScale>(&context);
        patterns2.add<MergeAluAct>(&context);

        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns2)))) {
            getOperation().emitError() << "pass failed";
            return signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createSimplifyGenericPass() {
    return std::make_unique<SimplifyGenericPass>();
}

} // namespace mlir::syna::torq_hl
