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
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-torq-softmax"

namespace mlir::syna::torq {

namespace {

Value makeElementWiseBinary(
    linalg::GenericOp srcOp, PatternRewriter &rewriter, Value input0, Value input1,
    torq_hl::ElementwiseOpEnum opType
) {
    auto resultType = dyn_cast<RankedTensorType>(input0.getType());

    return rewriter
        .create<torq_hl::ElementWiseBinaryOp>(
            srcOp.getLoc(), resultType, createInitTensor(srcOp, rewriter, resultType), opType,
            input0, input1, /*isUnsigned=*/false
        )
        .getOutput();
}

class SoftmaxOpPattern : public OpRewritePattern<linalg::SoftmaxOp> {
  public:
    SoftmaxOpPattern(MLIRContext *context)
        : OpRewritePattern<linalg::SoftmaxOp>(context, /*benefit=*/0) {
        setDebugName("SoftmaxOpPattern");
    }

    LogicalResult matchAndRewrite(linalg::SoftmaxOp op, PatternRewriter &rewriter) const override {

        // checks
        if (!cast<RankedTensorType>(op.getType(0)).getElementType().isBF16()) {
            return rewriter.notifyMatchFailure(op, "Expected bf16 output");
        }

        auto loc = op.getLoc();
        auto ctx = rewriter.getContext();
        auto input = op.getInput();
        auto bf16TType = dyn_cast<RankedTensorType>(input.getType());
        auto bf16 = bf16TType.getElementType();
        auto shape = bf16TType.getShape();
        auto rank = bf16TType.getRank();
        auto i16 = IntegerType::get(ctx, 16);
        auto i16TType = RankedTensorType::get(shape, i16);
        auto softmaxDim = op.getDimension();
        auto expandDim = rewriter.getDenseI64ArrayAttr({(long)(softmaxDim)});
        auto emptyLike = [&](Value val) {
            return rewriter.create<tensor::EmptyOp>(loc, val.getType(), ValueRange{});
        };

        SmallVector<AffineExpr> dims;
        SmallVector<AffineExpr> nonSoftmaxDims;
        SmallVector<int64_t> reduceShape;
        AffineExpr d;
        for (int i = 0; i < rank; i++) {
            d = getAffineDimExpr(i, ctx);
            dims.push_back(d);
            if (i != softmaxDim) {
                nonSoftmaxDims.push_back(d);
                reduceShape.push_back(shape[i]);
            }
        }
        SmallVector<AffineExpr> trailDim = {getAffineDimExpr(rank, ctx)};
        auto reduceBfTType = RankedTensorType::get(reduceShape, bf16);
        auto reduceITType = RankedTensorType::get(reduceShape, i16);

        auto negInf = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getFloatAttr(bf16, -std::numeric_limits<float>::infinity())
        );

        // subtract off bias
        auto maxAlloc =
            rewriter
                .create<linalg::FillOp>(
                    loc, ValueRange{negInf},
                    ValueRange{rewriter.create<tensor::EmptyOp>(loc, reduceBfTType, ValueRange{})}
                )
                .getResult(0);
        auto max = rewriter
                       .create<linalg::ReduceOp>(
                           loc, input, maxAlloc, softmaxDim,
                           [&](OpBuilder &b, Location l, ValueRange args) {
                               b.create<linalg::YieldOp>(
                                   l, ValueRange{b.create<arith::MaximumFOp>(l, args[0], args[1])}
                               );
                           }
                       )
                       .getResult(0);
        auto broadcastMax = rewriter
                                .create<torq_hl::BroadcastOp>(
                                    op.getLoc(), bf16TType,
                                    createInitTensor(op, rewriter, bf16TType), expandDim, max
                                )
                                .getOutput();
        auto identityMap = rewriter.getMultiDimIdentityMap(rank);
        auto biasedX = rewriter.create<linalg::GenericOp>(
            loc, TypeRange{bf16TType}, ValueRange{broadcastMax, input},
            ValueRange{emptyLike(input)},
            SmallVector<AffineMap>{identityMap, identityMap, identityMap},
            SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
            [&](OpBuilder &b, Location l, ValueRange args) {
                auto biased = b.create<arith::SubFOp>(l, args[1], args[0]);
                b.create<linalg::YieldOp>(l, ValueRange{biased});
            }
        );

        auto intBiasedX = makeBitcast(biasedX, rewriter, i16TType, biasedX.getResult(0));
        // auto signBit = rewriter.create<arith::ConstantOp>(
        //     input.getLoc(), RankedTensorType::get(ArrayRef<long>{1}, i16),
        //     DenseElementsAttr::get(
        //         RankedTensorType::get(ArrayRef<long>{1}, i16),
        //         {
        //             APInt(16, -(1 << 15)),
        //         }
        //     )
        // );
        // auto signBit = rewriter.create<arith::ConstantOp>(
        //     input.getLoc(), rewriter.getIntegerAttr(i16, -(1 << 15))
        // );
        auto signBit =
            rewriter
                .create<linalg::FillOp>(
                    loc,
                    ValueRange{rewriter.create<arith::ConstantOp>(
                        loc, rewriter.getIntegerAttr(i16, -(1 << 15))
                    )},
                    ValueRange{rewriter.create<tensor::EmptyOp>(loc, i16TType, ValueRange{})}
                )
                .getResult(0);
        auto adjustedBiasedX = makeElementWiseBinary(
            biasedX, rewriter, intBiasedX, signBit, torq_hl::ElementwiseOpEnum::BITWISE_OR
        );
        // ^ could be done with ui16->i16 rescale instead (i.e. engineering hours...)

        auto exp = makeScaledLut(
            biasedX, rewriter, adjustedBiasedX, 16256, 16, -32768, 0,
            SmallVector<int32_t>{
                -294914,   32762,     32762,     32762,     294905,    229369,    229369,
                163833,    32762,     32762,     32762,     294905,    294905,    229369,
                229369,    32762,     32762,     32762,     32762,     294905,    229369,
                229369,    32762,     32762,     32762,     32762,     294905,    229369,
                229369,    163833,    32762,     32762,     32762,     294905,    229369,
                229369,    229369,    32762,     32762,     32762,     294905,    294905,
                229369,    229369,    32762,     32762,     32762,     32762,     294905,
                -98309,    229365,    163829,    32758,     32758,     32758,     294901,
                229365,    229365,    229365,    32758,     32758,     32758,     294901,
                294901,    229365,    229365,    32758,     32758,     32758,     32758,
                294901,    229365,    229365,    163829,    -98313,    32754,     32754,
                294897,    229361,    229361,    229361,    32754,     32754,     32754,
                294897,    294897,    229361,    229361,    32754,     32754,     32754,
                -32782,    294893,    229357,    229357,    32750,     32750,     32750,
                32750,     294893,    229357,    229357,    229357,    32750,     32746,
                32746,     294889,    294889,    229353,    229353,    32746,     32746,
                -294934,   32742,     294885,    229349,    229349,    32742,     32742,
                32742,     -98329,    294881,    229345,    229345,    163809,    32738,
                32738,     32738,     294881,    -163870,   229341,    229341,    32734,
                32734,     32734,     -229410,   294873,    229337,    229337,    -32806,
                32726,     32726,     32726,     -98345,    229329,    229329,    163793,
                32722,     -294958,   32718,     294861,    229325,    -32818,    229321,
                32714,     32714,     -98357,    294853,    294853,    229317,    229317,
                -294970,   32706,     32706,     32706,     -32830,    229309,    -98369,
                163769,    32698,     -294982,   32694,     -163914,   229297,    -32846,
                229293,    -98385,    32682,     32682,     294821,    294821,    -98394,
                229281,    -98397,    32670,     32670,     32666,     294809,    229269,
                229269,    -32874,    32658,     -98413,    32654,     294797,    229257,
                229257,    -98422,    32646,     -295034,   -32894,    -98434,    -98438,
                229236,    229232,    -295055,   -32915,    -98454,    -98458,    294752,
                229212,    -98467,    -98470,    -98474,    32593,     32589,     -164019,
                -32951,    -98490,    229184,    -295103,   -32963,    -98502,    294708,
                294704,    -98511,    -32979,    32553,     32549,     -295131,   -32991,
                -98530,    -32999,    229136,    -98546,    -295159,   -33024,    -229636,
                -98571,    -98576,    -98583,    -98588,    -295200,   -98599,    -33068,
                -98611,    -33080,    -98620,    229051,    -98631,    -33100,    -295248,
                -98647,    -33116,    -98656,    229015,    -98667,    -33136,    -295284,
                -98683,    -98688,    -33157,    -98697,    163438,    32363,     -622998,
                -491936,   -33197,    -491954,   -426428,   -885186,   -491981,   -426453,
                -623068,   -33257,    -33265,    -33273,    -33282,    -1016330,  -492056,
                -950821,   -819765,   -950852,   -819796,   -688736,   -885359,   -623230,
                -1016458,  -492186,   -688806,   -885425,   -426690,   -885452,   -950999,
                -1016546,  -557808,   -557821,   -623368,   -1147668,  -1213224,  -1147711,
                -1344339,  -1213287,  -754560,   -951187,   -1147810,  -689079,   -1016777,
                -951257,   -1016808,  -885756,   -1999882,  -1934377,  -1541191,  -1672292,
                -2131069,  -1606810,  -1475765,  -1213648,  -1737956,  -1410304,  -1213720,
                -1475884,  -1148227,  -1082714,  -1148269,  -1082754,  -1213844,  -689579,
                -1082809,  -951757,   -2065879,  -2721271,  -3638820,  -3442271,  -3245719,
                -2721485,  -2721535,  -2918186,  -2262872,  -2525056,  -2066344,  -2131914,
                -1738735,  -3639307,  -3442760,  -3115139,  -3311797,  -3049704,  -2918680,
                -2132297,  -2656624,  -2328984,  -2132414,  -2001378,  -2656766,  -3639854,
                -3181164,  -2722469,  -2984662,  -3050245,  -2984756,  -2657119,  -2067340,
                -3574700,  -4295654,  -7506996,  -6720678,  -6065420,  -5082471,  -4558262,
                -6655472,  -7048794,  -6459079,  -5738281,  -4886400,  -4296653,  -7311368,
                -6656125,  -6066407,  -4821322,  -4559259,  -4166117,  -7639590,  -6591134,
                -5870341,  -5280608,  -4101046,  -4690936,  -7509064,  -6788282,  -5739809,
                -5215609,  -4494792,  -5019149,  -7181933,  -6461146,  -7509812,  -8624055,
                -13867087, -10721575, -10262979, -12819062, -10918719, -9673695,  -13606030,
                -10460507, -11902440, -12361397, -10067832, -11509769, -12558545, -9806224,
                -14066196, -11707123, -9479081,  -14656558, -11904271, -8758728,  -14394961,
                -11380524, -8562654,  -13412473, -10856780, -9218545,  -13609625, -10922852,
                -10791935, -12954811, -9678216,  -16494072, -20819747, -27373652, -23310803,
                -20165463, -23835818, -25212409, -20625288, -22722795, -25213493, -20757432,
                -21937446, -26656355, -23642073, -19513705, -24691374, -23184403, -19907988,
                -23709423, -25741371, -21481916, -21482292, -25873524, -22466021, -21876580,
                -26005677, -24105494, -20173732, -23713017, -26203711, -18470880, -23910702,
                -26335869, -21814267, -46194071, -49537097, -46523194, -50128362, -48425155,
                -46853026, -46329457, -46395709, -46789644, -46659292, -45480366, -50199649,
                -46530379, -46662163, -48235739, -46794680, -46991993, -46468442, -48107546,
                -46797548, -45356489, -47126670, -45095783, -46669353, -47915251, -43983839,
                -46409364, -49424731, -45689922, -48770807, -47067608, -46675112, -45234045,
                -99170851, -93863898, -96421241, -99633919, -96424117, -93541987, -92363790,
                -89088443, -93611842, -92040417, -91910792, -92829720, -27819560, -3702718,
                -360440
            },
            8482, 8, -17664, -32768
        );
        auto bfExp = makeBitcast(biasedX, rewriter, bf16TType, exp);
        auto denomAlloc =
            rewriter
                .create<linalg::FillOp>(
                    loc,
                    ValueRange{rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(bf16))},
                    ValueRange{rewriter.create<tensor::EmptyOp>(loc, reduceBfTType, ValueRange{})}
                )
                .getResult(0);
        auto denom = rewriter
                         .create<linalg::ReduceOp>(
                             loc, bfExp, denomAlloc, softmaxDim,
                             [&](OpBuilder &b, Location l, ValueRange args) {
                                 b.create<linalg::YieldOp>(
                                     l, ValueRange{b.create<arith::AddFOp>(l, args[0], args[1])}
                                 );
                             }
                         )
                         .getResult(0);
        auto denomI16 = makeBitcast(biasedX, rewriter, reduceITType, denom);
        auto recip = makeScaledLut(
            biasedX, rewriter, denomI16, 24576, 20, -32768, 14720,
            SmallVector<int32_t>{
                -16613397, -16613653, -16810474, -12616426, -12616640, -8422549,  -8619369,
                -12617237, -12617408, -8357781,  -8226880,  -12617920, -8620393,  -8358421,
                -8358549,  -8227648,  -8227776,  -8227904,  -8228032,  -8228160,  -8228288,
                -8359445,  -8359573,  -4230933,  -8490858,  -36842,    -4165696,  -8360085,
                -8491242,  -37226,    -37312,    -37397,    -37482,    -37568,    -37653,
                -8360768,  -4232085,  -37866,    -37952,    -4232298,  -38080,    -4232426,
                -38208,    -16750016, -12621504, -16881557, -8624277,  -8624490,  -16882197,
                -12622570, -8362944,  -8625258,  -8625429,  -8363456,  -8494698,  -12623594,
                -8363925,  -8626197,  -8364224,  -8364352,  -8364480,  -8364608,  -8364736,
                -4170560,  -8364949,  -8037400,  -8234092,  -42261,    -8234305,  -8037784,
                -108095,   -42645,    -42730,    -42816,    -42901,    -42986,    -43072,
                -108692,   -108777,   -43285,    -43370,    -108991,   -43498,    -109119,
                -43626,    -16427630, -16558957, -16428099, -8433003,  -8629823,  -8695570,
                -8499178,  -8433814,  -8630591,  -8172055,  -8237761,  -8631103,  -8500202,
                -8172695,  -8172823,  -8238529,  -8238657,  -8238785,  -8238913,  -8239041,
                -8239169,  -8173719,  -8239383,  -47552,    -8239596,  -113300,   -47850,
                -8239895,  -8239980,  -113684,   -113769,   -113855,   -113940,   -114025,
                -114111,   -8043970,  -48704,    -114324,   -114409,   -48917,    -114537,
                -49045,    -114665,   -16629783, -8634687,  -16433646, -8504085,  -8438763,
                -16434286, -8635754,  -8046146,  -8439531,  -8505237,  -8046658,  -8243436,
                -8636778,  -8243735,  -8506005,  -8047426,  -8047554,  -8047682,  -8047810,
                -8047938,  -52714,    -8244759,  -8179352,  -118591,   -8048494,  -8048579,
                -118889,   -53440,    -8048878,  -8048963,  -8180120,  -8049134,  -8049219,
                -8180376,  -54037,    -54122,    -119743,   -8049603,  -54336,    -119956,
                -54464,    -120084,   -8508693,  -16438725, -8378092,  -16439194, -16439408,
                -8444268,  -8444481,  -8510229,  -8510400,  -8445036,  -8445207,  -8510912,
                -8052334,  -8183576,  -8183704,  -8445975,  -8052931,  -8053059,  -8053187,
                -8053315,  -8053443,  -123753,   -8184728,  -8184856,  -8053870,  -58645,
                -8250690,  -8185240,  -124479,   -59029,    -59114,    -59200,    -59285,
                -59370,    -59456,    -125076,   -125161,   -59669,    -59754,    -125375,
                -59882,    -125503,   -60010,    -16575087, -16575343, -16444485, -8449388,
                -8646208,  -8711955,  -8515563,  -8450199,  -8646976,  -8188440,  -8254146,
                -8647488,  -8516587,  -8189080,  -8189208,  -8254914,  -8255042,  -8255170,
                -8255298,  -8255426,  -8255554,  -8190104,  -8255768,  -63936,    -8255981,
                -129684,   -64234,    -8256280,  -8256365,  -130068,   -130153,   -130239,
                -130324,   -130409,   -130495,   -8060355,  -65088,    -130708,   -130793,
                -65301,    -130921,   -65429,    -131049,   -16384026, -8585536,  -16581103,
                -8454934,  -8389612,  -16581743, -8586603,  -8193602,  -8390380,  -8456086,
                -8194114,  -8194285,  -8587627,  -7997977,  -8456854,  -8194882,  -8195010,
                -8195138,  -8195266,  -8195394,  -3562,     -7999001,  -7999130,  -69439,
                -8195950,  -7999428,  -69737,    -4288,     -8196334,  -7999812,  -7999898,
                -8196590,  -8000068,  -8000154,  -4885,     -4970,     -70591,    -8000452,
                -5184,     -70804,    -5312,     -70932,    -8459542,  -16389575, -8328941,
                -16390044, -16390258, -8395117,  -8395330,  -8461078,  -8461249,  -8395885,
                -8396056,  -8461761,  -8199790,  -8003354,  -8003482,  -8396824,  -8003780,
                -8003908,  -8004036,  -8004164,  -8004292,  -74601,    -8135577,  -8135705,
                -8201326,  -9493,     -8004932,  -8136089,  -75327,    -9877,     -9962,
                -10048,    -10133,    -10218,    -10304,    -75924,    -76009,    -10517,
                -10602,    -76223,    -10730,    -76351,    -10858,    -16394866, -16395122,
                -16526406, -8400237,  -8597057,  -8662804,  -8466412,  -8401048,  -8597825,
                -8139289,  -8008388,  -8598337,  -8467436,  -8139929,  -8140057,  -8009156,
                -8009284,  -8009412,  -8009540,  -8009668,  -8009796,  -8140953,  -8010010,
                -14784,    -8206830,  -80532,    -15082,    -8010522,  -8207214,  -80916,
                -81001,    -81087,    -81172,    -81257,    -81343,    -8207811,  -15936,
                -81556,    -81641,    -16149,    -81769,    -16277,    -81897,    -16400412,
                -8601921,  -16597489, -8471319,  -8405997,  -16598129, -8602988,  -8209987,
                -8406765,  -8472471,  -8210499,  -8210670,  -8604012,  -8014362,  -8473239,
                -8211267,  -8211395,  -8211523,  -8211651,  -8211779,  -19946,    -8015386,
                -8015515,  -85823,    -8212335,  -8015813,  -86121,    -20672,    -8212719,
                -8016197,  -8016283,  -8212975,  -8016453,  -8016539,  -21269,    -21354,
                -86975,    -8016837,  -21568,    -87188,    -21696,    -87316,    -8475927,
                -16405961, -8345326,  -16406430, -16406644, -8411502,  -8411715,  -8477463,
                -8477634,  -8412270,  -8412441,  -8478146,  -8019568,  -8019739,  -8019867,
                -8413209,  -8216772,  -8216900,  -8217028,  -8217156,  -8217284,  -90985,
                -8020891,  -8021019,  -8217711,  -25877,    -8217924,  -8021403,  -91711,
                -26261,    -26346,    -26432,    -26517,    -26602,    -26688,    -92308,
                -92393,    -26901,    -26986,    -92607,    -27114,    -92735,    -27242,
                -16411252, -16411508, -16411721, -8416622,  -8613442,  -8679189,  -8482797,
                -8417433,  -8614210,  -8024603,  -8221380,  -8614722,  -8483821,  -8025243,
                -8025371,  -8222148,  -8222276,  -8222404,  -8222532,  -8222660,  -8222788,
                -8026267,  -8223002,  -31168,    -8026608,  -96916,    -31466,    -8223514,
                -8026992,  -97300,    -97385,    -97471,    -97556,    -97641,    -97727,
                -8224196,  -32320,    -97940,    -98025,    -32533,    -98153,    -32661,
                -98281
            },
            10922, 8, 16256, -32768
        );
        auto bfRecip = makeBitcast(biasedX, rewriter, reduceBfTType, recip);
        // inefficient
        auto broadcastBfRecip =
            rewriter
                .create<torq_hl::BroadcastOp>(
                    op.getLoc(), bf16TType, createInitTensor(op, rewriter, bf16TType), expandDim,
                    bfRecip
                )
                .getOutput();
        auto softmax =
            rewriter
                .create<linalg::GenericOp>(
                    loc, TypeRange{bf16TType}, ValueRange{broadcastBfRecip, bfExp},
                    ValueRange{emptyLike(bfExp)},
                    SmallVector<AffineMap>{identityMap, identityMap, identityMap},
                    SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                    [&](OpBuilder &b, Location l, ValueRange args) {
                        auto result = b.create<arith::MulFOp>(l, args[0], args[1]);
                        b.create<linalg::YieldOp>(l, ValueRange{result});
                    }
                )
                .getResult(0);
        rewriter.replaceOp(op, softmax);
        return success();
    }
};

} // namespace

void populateSoftmaxPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.insert<SoftmaxOpPattern>(context);
}

} // namespace mlir::syna::torq
