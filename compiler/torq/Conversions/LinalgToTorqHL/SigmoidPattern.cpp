// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-torq-sigmoid"

namespace mlir::syna::torq {

namespace {

Value makeSelect(
    linalg::GenericOp srcOp, PatternRewriter &rewriter, Value pred, Value ifTrue, Value ifFalse
) {
    auto resultType = dyn_cast<RankedTensorType>(ifTrue.getType());
    return torq_hl::SelectOp::create(
               rewriter, srcOp.getLoc(), resultType, createInitTensor(srcOp, rewriter, resultType),
               pred, ifTrue, ifFalse
    )
        .getOutput();
}

class SigmoidOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    SigmoidOpPattern(MLIRContext *context)
        : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/0) {
        setDebugName("SigmoidOpPattern");
    }

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {

        auto yieldOp = dyn_cast<linalg::YieldOp>(srcOp.getBody()->getTerminator());
        if (!yieldOp) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected a linalg.yield terminator for SigmoidOpPattern"
            );
        }

        auto yieldValues = yieldOp.getValues();
        if (yieldValues.size() != 1) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected exactly one yield value for SigmoidOpPattern"
            );
        }

        auto errMsg = "Expected defining operations for yield operand to be arith.negf, math.exp, "
                      "arith.addf, and arith.divf.";

        auto divfOp = yieldValues[0].getDefiningOp<arith::DivFOp>();
        if (!divfOp)
            return rewriter.notifyMatchFailure(srcOp, errMsg);

        auto addfOp = divfOp.getRhs().getDefiningOp<arith::AddFOp>();
        if (!addfOp)
            return rewriter.notifyMatchFailure(srcOp, errMsg);

        auto expOp = addfOp.getLhs().getDefiningOp<math::ExpOp>();
        if (!expOp)
            return rewriter.notifyMatchFailure(srcOp, errMsg);

        auto negfOp = expOp.getOperand().getDefiningOp<arith::NegFOp>();
        if (!negfOp)
            return rewriter.notifyMatchFailure(srcOp, errMsg);

        auto input = srcOp.getInputs()[0];
        auto tType = dyn_cast<RankedTensorType>(input.getType());
        auto eType = tType.getElementType();

        if (!eType.isBF16()) {
            return rewriter.notifyMatchFailure(srcOp, "Expected bf16 output");
        }

        auto shape = tType.getShape();
        auto i16 = rewriter.getIntegerType(16);
        auto i1 = rewriter.getIntegerType(1);
        auto bf16Threshold = RankedTensorType::get(shape, eType);

        /// FIXME: temporary board-side mitigation until the high-positive Sigmoid
        /// LUT path is fully debugged and fixed at the real source.
        auto ones = arith::ConstantOp::create(
            rewriter, srcOp.getLoc(), bf16Threshold,
            DenseElementsAttr::get(bf16Threshold, rewriter.getFloatAttr(eType, 1.0))
        );
        auto positiveSaturationThreshold = arith::ConstantOp::create(
            rewriter, srcOp.getLoc(), bf16Threshold,
            DenseElementsAttr::get(bf16Threshold, rewriter.getFloatAttr(eType, 6.25))
        );

        auto x = makeBitcast(srcOp, rewriter, RankedTensorType::get(shape, i16), input);

        auto msb = arith::ConstantOp::create(
            rewriter, x.getLoc(), RankedTensorType::get(shape, i16),
            DenseElementsAttr::get(
                RankedTensorType::get(shape, i16),
                {
                    APInt(16, (1 << 15) - 1),
                }
            )
        );
        auto isNegative = torq_hl::ElementWiseBinaryOp::create(
                              rewriter, srcOp.getLoc(), RankedTensorType::get(shape, i1),
                              createInitTensor(srcOp, rewriter, RankedTensorType::get(shape, i1)),
                              torq_hl::ElementwiseOpEnum::GREATER, x, msb, /*isUnsigned=*/true
        )
                              .getOutput();

        auto sigmoidNegative = makeScaledLut(
            srcOp, rewriter, x, 16128, 16, -32768, 0,
            SmallVector<int32_t>{
                -294914,   32762,     229369,    32762,     229369,    32762,     32762,
                229369,    32762,     229369,    32762,     32762,     229369,    32762,
                229369,    32762,     294905,    229369,    32762,     229369,    32762,
                294905,    229369,    32762,     229369,    32762,     294905,    229369,
                32762,     229369,    32762,     294905,    229369,    32762,     229369,
                32762,     294905,    229369,    32762,     229369,    32762,     294905,
                163833,    32762,     229369,    32762,     294905,    163833,    32762,
                229369,    32762,     294905,    163833,    -32774,    229365,    32758,
                294901,    163829,    32758,     229365,    32758,     294901,    163829,
                32758,     229365,    32758,     294901,    163829,    32758,     229365,
                32758,     294901,    163829,    32758,     229365,    32758,     294901,
                163829,    32758,     229365,    -294922,   294897,    32754,     32754,
                229361,    32754,     294897,    32754,     32754,     229361,    32754,
                229361,    32754,     32754,     229361,    32754,     229361,    32754,
                32750,     229357,    32750,     229357,    32750,     32750,     229357,
                32750,     229357,    32750,     294893,    229357,    32750,     -98322,
                32746,     294889,    229353,    32746,     229353,    32746,     294889,
                229353,    -229398,   229349,    32742,     294885,    229349,    32742,
                229349,    32742,     294885,    163809,    32738,     229345,    32738,
                294881,    163809,    32738,     229345,    32738,     294877,    163805,
                32734,     229341,    32734,     294877,    -98337,    32730,     229337,
                32730,     294873,    163797,    32726,     229333,    32726,     -32810,
                163793,    32722,     229329,    32722,     294861,    163789,    32718,
                229325,    -32818,    294857,    163785,    32714,     229321,    32710,
                294853,    32710,     32710,     -98362,    32706,     294849,    32706,
                -98365,    229308,    32701,     229308,    -294979,   32697,     -98375,
                32693,     -32843,    32689,     -98382,    229292,    32685,     229288,
                32681,     -229463,   229284,    -32859,    229280,    -98398,    294812,
                229276,    32665,     229272,    -295015,   294804,    -98411,    32657,
                -32879,    32653,     -98418,    229256,    32649,     -98423,    32645,
                -32891,    229247,    -98431,    229243,    -295044,   -163976,   -98444,
                -32912,    229227,    32616,     294755,    163679,    -229536,   -98468,
                -32936,    -32940,    -98479,    -98483,    229191,    32580,     -164028,
                -32961,    -32965,    -32969,    -98508,    -98512,    163626,    -229589,
                -98521,    -32989,    -32993,    -98532,    -98536,    229138,    -295149,
                -164081,   -33013,    -33017,    229122,    -33026,    -33034,    -33038,
                -33046,    -33054,    -295202,   -33066,    -98609,    -229686,   -33086,
                -426307,   -164171,   -491855,   -491862,   -33119,    -426339,   -98667,
                -491887,   -229751,   -33152,    -295300,   -33164,    -295312,   -33176,
                228959,    -33188,    228947,    -33200,    -491955,   -98747,    -426433,
                -33225,    -295373,   -33237,    -98777,    -98784,    -98793,    -885233,
                -754172,   -1016330,  -1212957,  -1016371,  -1344071,  -1016413,  -623221,
                -950919,   -1147545,  -1213100,  -1147583,  -623319,   -1213156,  -1147638,
                -1147657,  -951068,   -623408,   -1147709,  -754513,   -1016670,  -623474,
                -623490,   -885647,   -754590,   -885676,   -951226,   -820170,   -885720,
                -754663,   -558069,   -623617,   -1082382,  -1410084,  -2982969,  -2917481,
                -3048596,  -2589891,  -2589933,  -2065688,  -2393405,  -1934691,  -1541514,
                -1869225,  -2065862,  -1869288,  -1934849,  -3048998,  -3180122,  -2590352,
                -3180218,  -2655979,  -1935130,  -2197312,  -2131815,  -2066315,  -2066350,
                -1738706,  -2066413,  -952339,   -3246131,  -3704934,  -2918556,  -3115212,
                -2722045,  -2525480,  -2525521,  -2460023,  -2328988,  -2001346,  -3967465,
                -6982180,  -6261395,  -5278457,  -5147467,  -4426652,  -4230112,  -6392858,
                -6327429,  -5672169,  -5344578,  -4492692,  -4099549,  -5475865,  -6786676,
                -6262490,  -5017404,  -4689807,  -4231127,  -3969051,  -6918248,  -6066388,
                -5673265,  -5083526,  -3969494,  -4231701,  -6853215,  -6263499,  -5608235,
                -4952962,  -4232144,  -3642392,  -7312464,  -6329536,  -5608741,  -8230259,
                -7837179,  -13014675, -9803614,  -8361974,  -12556428, -10197333, -8165877,
                -13081216, -10263375, -8297458,  -13671538, -10788166, -8297967,  -13803114,
                -11181883, -8822759,  -13475940, -11116854, -8692196,  -13148765, -11313968,
                -9020384,  -13149270, -11904291, -9086427,  -11511383, -12166937, -9218007,
                -10922073, -12167443, -9742798,  -10070619, -12102413, -9874376,  -9612376,
                -21671199, -23637587, -19443655, -22851848, -22131276, -18985923, -24360174,
                -20362823, -19773360, -24754393, -19839554, -20691875, -25738438, -19971644,
                -21086101, -27246760, -19448374, -21873544, -23119035, -19318319, -22660979,
                -22137010, -19122729, -24234840, -22269095, -18533925, -24891203, -20304038,
                -20369932, -25088819, -19715235, -20829696, -26793754, -19912858, -21223922,
                -36231900, -44227876, -39903210, -43770488, -44360969, -40167383, -43641441,
                -46525678, -45019028, -43840070, -45872345, -44300159, -41744957, -43318487,
                -44105580, -40698414, -43386046, -45680465, -43846134, -43977888, -45879092,
                -44241372, -41030816, -43587378, -44243403, -39918738, -42671907, -46604710,
                -44246098, -43198213, -44640668, -44313660, -41037564, -43921804, -44250152,
                -39728880, -87046452, -86851191, -86983622, -86657286, -90066485, -87053202,
                -88103116, -86990372, -87581534, -89417878, -87256558, -88437545, -87193724,
                -33521155
            },
            9108, 8, -17536, -32768
        );

        auto sigmoidPositive = makeScaledLut(
            srcOp, rewriter, x, 2048, 20, -32768, 16128,
            SmallVector<int32_t>{
                39747584, 164095,   33024,    164095,   229631,   33024,    164095,   33024,
                164095,   229631,   33024,    164095,   33024,    164095,   229631,   33024,
                164095,   33024,    164095,   229631,   33024,    164095,   229631,   33024,
                164095,   33024,    164095,   229631,   33024,    164095,   33024,    164095,
                229631,   33024,    164095,   33024,    164095,   229631,   33024,    164095,
                33024,    164095,   164095,   33024,    164095,   229631,   33024,    164095,
                33024,    164095,   229631,   33024,    164095,   33024,    164095,   229631,
                33024,    164095,   33024,    164095,   229631,   33024,    164095,   33024,
                33024,    164095,   33024,    164095,   229631,   33024,    164095,   33024,
                164095,   229631,   33024,    164095,   33024,    164095,   229631,   33024,
                79396864, 33536,    164607,   230143,   33536,    164607,   230143,   33536,
                164607,   33536,    164607,   230143,   33536,    164607,   33536,    164607,
                230143,   33536,    164607,   33536,    164607,   230143,   33536,    164607,
                33536,    164607,   164607,   33536,    164607,   230143,   33536,    164607,
                33536,    164607,   230143,   33536,    164607,   33536,    164607,   230143,
                79725245, 165119,   34048,    165119,   230655,   34048,    165119,   34048,
                34048,    165119,   34048,    165119,   230655,   34048,    165119,   34048,
                165119,   230655,   34048,    165119,   34048,    165119,   230655,   34048,
                165119,   34048,    165119,   79725529, 34560,    165631,   231167,   34560,
                165631,   34560,    165631,   231167,   34560,    165631,   34560,    165631,
                231167,   34560,    165631,   34560,    165631,   231167,   34560,    79725946,
                35072,    166143,   166143,   35072,    166143,   231679,   35072,    166143,
                35072,    166143,   231679,   35072,    166143,   35584,    166655,   232191,
                35584,    166655,   35584,    166655,   232191,   35584,    166655,   35584,
                35584,    166655,   79399577, 167167,   232703,   36096,    167167,   36096,
                167167,   232703,   36096,    167167,   36096,    167167,   232703,   79203508,
                167679,   36608,    167679,   233215,   36608,    167679,   233215,   36608,
                167679,   79728260, 168191,   233727,   37120,    168191,   37120,    168191,
                79728639, 37632,    168703,   37632,    168703,   234239,   37632,    169215,
                38144,    169215,   169215,   38144,    169215,   234751,   38656,    169727,
                38656,    169727,   235263,   38656,    79730080, 39168,    170239,   235775,
                39168,    170239,   39168,    170751,   236287,   39680,    170751,   39680,
                39680,    170751,   79600270, 171263,   236799,   40192,    171263,   40192,
                171263,   237311,   40704,    171775,   40704,    81173788, 237823,   41216,
                172287,   41728,    172799,   238335,   79340193, 173311,   238847,   42240,
                173823,   42752,    173823,   81110542, 43264,    174335,   79210610, 174847,
                240383,   43776,    175359,   44288,    175359,   240895,   44800,    175871,
                44800,    79474442, 176383,   45312,    81178454, 242431,   45824,    176895,
                79409798, 177407,   242943,   79738064, 177919,   46848,    177919,   243967,
                47360,    178431,   79738955, 178943,   244479,   47872,    179455,   48384,
                48384,    79739846, 48896,    180479,   246015,   79544557, 180991,   50432,
                79741761, 247551,   78104273, 182527,   79743070, 183039,   249087,   52480,
                184063,   79547970, 184575,   250111,   54016,    79745383, 251135,   71357687,
                186111,   79484559, 186623,   81123342, 56064,    187647,   56576,    188159,
                253695,   79552199, 188671,   58112,    189183,   255231,   58624,    190207,
                59136,    79488778, 190719,   60160,    191231,   257279,   60672,    192255,
                61184,    192767,   258303,   79556826, 193279,   79622760, 193791,   259839,
                79754941, 81131389, 79493718, 79756059, 81133012, 79298721, 131839,   1280,
                79496448, 133375,   79562894, 81201484, 199935,   3840,     135423,   79433852,
                79696193, 201983,   5888,     137471,   79698014, 81139996, 204031,   79502516,
                81141107, 79503426, 140031,   79700441, 79701143, 141055,   79570479, 71313655,
                142079,   79702660, 81079107, 208639,   79441636, 143615,   13056,    79638840,
                210175,   79508679, 145151,   14592,    145663,   211711,   79706794, 146687,
                81017909, 16128,    79576528, 16640,    79642462, 79708709, 79512813, 81151405,
                79448187, 81152313, 216319,   20224,    151807,   79581288, 79712558, 218367,
                79713469, 81089917, 22784,    154367,   219903,   23808,    79649649, 24320,
                79519488, 155903,   79585422, 156415,   222463,   25856,    81093540, 26368,
                79717697, 223487,   26880,    158463,   27392,    158463,   224511,   27904,
                158975,   28416,    159487,   225023,   79720087, 159999,   225535,   28928,
                79654845, 29440,    160511,   226047,   29440,    161023,   29952,    161023,
                226559,   29952,    161535,   30464,    161535,   81098206, 30976,    162047,
                30976,    30976,    162559,   31488,    162559,   228095,   31488,    162559,
                31488,    163071,   228607,   32000,    163071,   32000,    163071,   228607,
                32000,    163071,   32000,    163071,   228607,   32000,    163071,   80706604
            },
            13707, 8, 15360, -32768
        );

        auto rawSigmoid = makeSelect(srcOp, rewriter, isNegative, sigmoidNegative, sigmoidPositive);
        auto sigmoid = makeBitcast(srcOp, rewriter, tType, rawSigmoid);

        /// FIXME: remove this saturation guard once the underlying high-positive
        /// Sigmoid miscompute is fixed.
        auto isSaturatedPositive =
            torq_hl::ElementWiseBinaryOp::create(
                rewriter, srcOp.getLoc(), RankedTensorType::get(shape, i1),
                createInitTensor(srcOp, rewriter, RankedTensorType::get(shape, i1)),
                torq_hl::ElementwiseOpEnum::GREATER_EQUAL, input, positiveSaturationThreshold,
                /*isUnsigned=*/false
            )
                .getOutput();
        auto saturatedSigmoid = makeSelect(srcOp, rewriter, isSaturatedPositive, ones, sigmoid);
        rewriter.replaceOp(srcOp, saturatedSigmoid);
        return success();
    }
};

} // namespace

void populateSigmoidPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.insert<SigmoidOpPattern>(context);
}

} // namespace mlir::syna::torq
