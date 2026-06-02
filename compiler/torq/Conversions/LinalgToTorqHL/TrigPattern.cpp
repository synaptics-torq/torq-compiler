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

#define DEBUG_TYPE "linalg-torq-trig"

namespace mlir::syna::torq {

namespace {

class TrigOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    TrigOpPattern(MLIRContext *context)
        : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/0) {
        setDebugName("GenericOpPattern");
    }

    std::tuple<Value, Value, Value>
    rangeReduce(linalg::GenericOp op, PatternRewriter &rewriter, Value input) const {
        auto fpConst = [&](double fp) {
            return arith::ConstantOp::create(
                rewriter, op.getLoc(), rewriter.getFloatAttr(rewriter.getBF16Type(), fp)
            );
        };
        auto pi = fpConst(M_PI);
        auto piOverTwo = fpConst(M_PI / 2);
        auto twoPi = fpConst(2 * M_PI);
        auto oneOverTwoPi = fpConst(1 / (2 * M_PI));

        auto resultType = cast<RankedTensorType>(op.getType(0));
        auto rank = resultType.getRank();
        auto ctx = rewriter.getContext();

        auto broadcast = [&](Value v, auto func) {
            return linalg::GenericOp::create(
                       rewriter, op.getLoc(), TypeRange{resultType}, ValueRange{v},
                       ValueRange{
                           tensor::EmptyOp::create(rewriter, op.getLoc(), resultType, ValueRange{})
                       },
                       op.getIndexingMapsArray(), op.getIteratorTypesArray(),
                       [&](OpBuilder &b, Location l, ValueRange args) {
                           linalg::YieldOp::create(b, l, ValueRange{func(b, l, args)});
                       }
            ).getResult(0);
        };
        auto broadcastBinary = [&](Value v, Value w, auto func) {
            return linalg::GenericOp::create(
                       rewriter, op.getLoc(), TypeRange{resultType}, ValueRange{v, w},
                       ValueRange{
                           tensor::EmptyOp::create(rewriter, op.getLoc(), resultType, ValueRange{})
                       },
                       SmallVector<AffineMap>(3, AffineMap::getMultiDimIdentityMap(rank, ctx)),
                       SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                       [&](OpBuilder &b, Location l, ValueRange args) {
                           linalg::YieldOp::create(b, l, ValueRange{func(b, l, args)});
                       }
            ).getResult(0);
        };
        auto gt = [&](Value tensor, Value scalar) {
            auto emptyTensor =
                tensor::EmptyOp::create(rewriter, op.getLoc(), resultType, ValueRange{});
            auto filled = linalg::FillOp::create(
                              rewriter, op.getLoc(), ValueRange{scalar}, ValueRange{emptyTensor}
            )
                              .getResult(0);
            return makeElementWiseBinary(
                op, rewriter, tensor, filled, torq_hl::ElementwiseOpEnum::GREATER
            );
        };
        auto select = [&](Value b, Value ifTrue, Value ifFalse) {
            return makeSelect(op, rewriter, b, ifTrue, ifFalse);
        };

        // range reduce down to 2pi
        auto branchLoc = broadcast(input, [&](OpBuilder &b, Location l, ValueRange args) {
            return arith::MulFOp::create(b, l, args[0], oneOverTwoPi);
        });
        auto branchNum =
            torq_hl::ActOp::create(
                rewriter, op.getLoc(), resultType, createInitTensor(op, rewriter, resultType),
                "floor", 0, 0, 0, 0, APFloat(llvm::APFloat::IEEEsingle(), "0.0"),
                APFloat(llvm::APFloat::IEEEsingle(), "0.0"), branchLoc,
                /*weights=*/mlir::Value()
            )
                .getOutput();
        auto branchBase = broadcast(branchNum, [&](OpBuilder &b, Location l, ValueRange args) {
            return arith::MulFOp::create(b, l, args[0], twoPi);
        });
        auto twoPiReduced =
            broadcastBinary(input, branchBase, [&](OpBuilder &b, Location l, ValueRange args) {
                return arith::SubFOp::create(b, l, args[0], args[1]);
            });

        // range reduce down to quadrants 1 and 2
        auto q34 = gt(twoPiReduced, pi);
        auto reflectVertical =
            broadcast(twoPiReduced, [&](OpBuilder &b, Location l, ValueRange args) {
                return arith::SubFOp::create(b, l, twoPi, args[0]);
            });
        auto piReduced = select(q34, reflectVertical, twoPiReduced);

        // range reduce down to quadrant 1
        auto q23 = gt(piReduced, piOverTwo);
        auto reflectHorizontal =
            broadcast(piReduced, [&](OpBuilder &b, Location l, ValueRange args) {
                return arith::SubFOp::create(b, l, pi, args[0]);
            });
        auto reduced = select(q23, reflectHorizontal, piReduced);

        return {reduced, q34, q23};
    }

    Value buildCos(linalg::GenericOp op, PatternRewriter &rewriter, Value input) const {
        return makeScaledLut(
            op, rewriter, input, 22560, 20, -32768, 14846,
            SmallVector<int32_t>{-3440663,   32721,      32721, 32721, 32721, 32721, 32721,
                                 32699,      32721,      32721, 32721, 32721, 32721, 32721,
                                 32699,      32721,      32721, 32721, 32721, 32721, 32721,
                                 32699,      32721,      32721, 32721, 32721, 32721, 32721,
                                 32699,      32721,      32721, 32721, 32721, 32721, 32721,
                                 32699,      32721,      32721, 32721, 32721, 32721, 32721,
                                 32699,      32721,      32721, 32721, 32721, 32721, 32721,
                                 32699,      32721,      32721, 32721, 32721, 32721, 32721,
                                 32699,      32721,      32721, 32721, 32721, 32721, 32721,
                                 32699,      32721,      32721, 32721, 32721, 32721, 32721,
                                 32699,      32721,      32721, 32721, 32721, 32721, 32721,
                                 32699,      32721,      32721, 32721, 32721, 32674, 32674,
                                 32652,      32674,      32674, 32674, 32674, 32674, 32674,
                                 32652,      32674,      32674, 32674, 32674, 32674, 32674,
                                 32652,      32674,      32674, 32674, 32674, 32674, 32674,
                                 32652,      32674,      32674, 32674, 32674, 32674, 32674,
                                 32652,      32674,      32674, 32674, 32674, 32674, 32674,
                                 32652,      32674,      32674, 32674, 32674, 32674, 32674,
                                 32606,      32628,      32628, 32628, 32628, 32628, 32628,
                                 32606,      32628,      32628, 32628, 32628, 32628, 32628,
                                 32606,      32628,      32628, 32628, 32628, 32628, 32628,
                                 32606,      32581,      32581, 32581, 32581, 32581, 32581,
                                 32559,      32581,      32581, 32581, 32581, 32581, 32581,
                                 32559,      32581,      32581, 32581, 32581, 32581, 32581,
                                 -3440847,   32535,      32535, 32535, 32535, 32535, 32535,
                                 32513,      32535,      32535, 32535, 32535, 32535, 32535,
                                 32513,      32535,      32535, 32535, 32488, 32488, 32488,
                                 32466,      32488,      32488, 32488, 32488, 32488, 32488,
                                 32466,      32488,      32488, 32488, 32488, 32488, 32488,
                                 32420,      32442,      32442, 32442, 32442, 32442, 32442,
                                 32420,      32442,      32442, 32442, 32442, 32442, 32442,
                                 -3506521,   32395,      32395, 32395, 32395, 32395, 32395,
                                 32373,      32395,      32395, 32395, 32349, 32349, 32349,
                                 32327,      32349,      32349, 32302, 32302, 32302, 32302,
                                 32280,      32302,      32302, 32256, 32256, 32256, 32256,
                                 32234,      32256,      32209, 32209, 32209, 32209, 32209,
                                 -3441217,   32163,      32163, 32163, 32163, 32163, 32163,
                                 32095,      32116,      32116, 32116, 32116, 32070, 32070,
                                 32048,      32070,      32070, 32023, 32023, 32023, 32023,
                                 -3506938,   31977,      31977, 31977, 31977, 31977, 31930,
                                 31909,      31930,      31930, 31884, 31884, 31884, 31884,
                                 31862,      31837,      31837, 31837, 31837, 31837, 31791,
                                 31769,      31791,      31791, 31744, 31744, 31744, 31744,
                                 -3507217,   31698,      31698, 31698, 31698, 31652, 31652,
                                 31630,      31605,      31605, 31605, 31605, 31605, 31559,
                                 31537,      31559,      31559, 31512, 31512, 31512, 31512,
                                 31444,      31466,      31466, 31419, 31419, 31419, 31419,
                                 -3376471,   31373,      31373, 31373, 31326, 31326, 31326,
                                 -3376564,   31280,      31233, 31233, 31187, 31187, 31140,
                                 -3376750,   31094,      31047, 31047, 31001, 31001, 30954,
                                 -3376935,   30861,      30861, 30815, 30815, 30768, 30768,
                                 30700,      30675,      30629, 30629, 30583, 30583, 30536,
                                 -3377354,   30443,      30443, 30397, 30397, 30350, 30304,
                                 -3508657,   30257,      30211, 30164, 30164, 30118, 30071,
                                 -3443354,   30025,      29978, 29932, 29932, 29885, 29839,
                                 -3443586,   29746,      29746, 29699, 29653, 29653, 29606,
                                 29538,      29513,      29467, 29467, 29421, 29374, 29328,
                                 -3509633,   29235,      29235, 29188, 29142, 29095, 29095,
                                 -3444376,   29002,      28956, 28909, 28863, 28863, 28816,
                                 -3379119,   28723,      28677, 28630, 28584, 28584, 28537,
                                 -3379398,   28444,      28398, 28352, 28305, 28259, 28259,
                                 -3379677,   28119,      28119, 28073, 28026, 27980, 27933,
                                 -3445491,   27840,      27794, 27747, 27701, 27701, 27654,
                                 -3445816,   27515,      27468, 27468, 27422, 27375, 27329,
                                 -3511677,   27143,      27050, 27004, 26911, 26818, 26678,
                                 -10459272,  26167,      26028, 25842, 25702, 25516, 25330,
                                 -13868440,  24866,      24680, 24494, 24354, 24168, 23983,
                                 -10461966,  23471,      23285, 23146, 22960, 22774, 22588,
                                 -10266754,  22077,      21891, 21705, 21519, 21380, 21194,
                                 -17346023,  20404,      20032, 19706, 19335, 18963, 18591,
                                 -27834553,  17476,      17150, 16778, 16406, 16035, 15663,
                                 15291,      -55690703,  13478, 12735, 11991, 11247, 10504,
                                 9760,       -107928798, 6088,  4601,  3114,  232,   62793,
                                 -1833374611},
            28679, 8, 15744, -32768
        );
    }
    Value buildSin(linalg::GenericOp op, PatternRewriter &rewriter, Value input) const {
        return makeScaledLut(
            op, rewriter, input, 16256, 16, -32768, 0,
            SmallVector<int32_t>{
                8355840, 8355968, 8421632, 8356225, 8421889, 8356482, 8487682, 8356739, 8422403,
                8356996, 8488196, 8422789, 8488453, 8423046, 8488710, 8423303, 8488967, 8423560,
                8489224, 8423817, 8489481, 8424074, 8489738, 8424331, 8489995, 8424588, 8490252,
                8490381, 8490509, 8490638, 8425231, 8490895, 8425488, 8491152, 8425745, 8491409,
                8426002, 8491666, 8426259, 8491923, 8426516, 8492180, 8426773, 8492437, 8427030,
                8492694, 8427287, 8492951, 8427544, 8493208, 8427801, 8493465, 8493594, 8493722,
                8493851, 8493979, 8494108, 8494236, 8494365, 8428958, 8494622, 8429215, 8494879,
                8429472, 8495136, 8429729, 8495393, 8429986, 8495650, 8430243, 8495907, 8430500,
                8496164, 8430757, 8496421, 8431014, 8496678, 8431271, 8496935, 8497064, 8497192,
                8497321, 8497449, 8497578, 8497706, 8497835, 8497963, 8498092, 8432685, 8498349,
                8432942, 8498606, 8433199, 8498863, 8433456, 8499120, 8433713, 8499377, 8433970,
                8499634, 8434227, 8499891, 8434484, 8500148, 8500277, 8500405, 8500534, 8500662,
                8500791, 8500919, 8501048, 8501176, 8501305, 8501433, 8501562, 8501690, 8501819,
                8436412, 8502076, 8436669, 8502333, 8436926, 8502590, 8437183, 8502847, 8437440,
                8503104, 8437697, 8503361, 8503490, 8569154, 8569283, 8503875, 8504004, 8569668,
                8569797, 8569925, 8570054, 8570182, 8570311, 8570439, 8570568, 8570696, 8570825,
                8439882, 8571082, 8440139, 8571339, 8440396, 8571596, 8440653, 8571853, 8440910,
                8572110, 8441167, 8572367, 8572496, 8572624, 8572753, 8572881, 8573010, 8573138,
                8573267, 8573395, 8573524, 8573652, 8573781, 8573909, 8574038, 8574166, 8574295,
                8574423, 8574552, 8443609, 8574809, 8443866, 8575066, 8444123, 8575323, 8444380,
                8575580, 8575709, 8575837, 8575966, 8576094, 8576223, 8576351, 8576480, 8576608,
                8576737, 8576865, 8576994, 8577122, 8577251, 8642915, 8643044, 8643172, 8643301,
                8643429, 8643558, 8643686, 8643815, 8447336, 8644072, 8447593, 8644329, 8644458,
                8644586, 8644715, 8644843, 8382829, 8645100, 8645229, 8645357, 8383343, 8711150,
                8449136, 8711407, 8449393, 8711664, 8449650, 8711921, 8449907, 8712178, 8450164,
                8712435, 8450421, 8450549, 8450678, 8450806, 8450935, 8451063, 8451192, 8385785,
                8451449, 8386042, 8451706, 8386299, 8451963, 8386556, 8452220, 8386813, 8452477,
                8387070, 8452734, 8452863, 8452991, 8453120, 8453248, 8453377, 8453505, 8453634,
                8453762, 8453891, 8519555, 8388612, 8388740, 8454405, 8388998, 8454662, 8389255,
                8454919, 8389512, 8455176, 8389769, 8455433, 8390026, 8455690, 8390283, 8455947,
                8390540, 8456204, 8390797, 8456461, 8391054, 8456718, 8391311, 8456975, 8391568,
                8457232, 8391825, 8457489, 8392082, 8457746, 8457875, 8458003, 8458132, 8458260,
                8458389, 8392982, 8458646, 8393239, 8458903, 8393496, 8459160, 8393753, 8459417,
                8394010, 8459674, 8394267, 8459931, 8394524, 8460188, 8394781, 8460445, 8395038,
                8460702, 8395295, 8460959, 8461088, 8461216, 8461345, 8461473, 8461602, 8461730,
                8461859, 8396452, 8462116, 8396709, 8462373, 8396966, 8462630, 8397223, 8462887,
                8397480, 8463144, 8397737, 8463401, 8397994, 8463658, 8398251, 8463915, 8398508,
                8464172, 8464301, 8464429, 8464558, 8464686, 8464815, 8464943, 8465072, 8465200,
                8465329, 8465457, 8465586, 8400179, 8465843, 8400436, 8466100, 8400693, 8466357,
                8400950, 8466614, 8401207, 8466871, 8401464, 8467128, 8401721, 8467385, 8401978,
                8467642, 8467771, 8467899, 8468028, 8468156, 8468285, 8468413, 8468542, 8468670,
                8468799, 8468927, 8469056, 8469184, 8469313, 8403906, 8469570, 8404163, 8535363,
                8404420, 8535620, 8404677, 8535877, 8404934, 8536134, 8405191, 8536391, 8536520,
                8536648, 8536777, 8536905, 8537034, 8537162, 8537291, 8537419, 8537548, 8537676,
                8537805, 8537933, 8538062, 8538190, 8538319, 8538447, 8538576, 8407633, 8538833,
                8407890, 8539090, 8408147, 8539347, 8408404, 8539604, 8408661, 8539861, 8539990,
                8540118, 8540247, 8540375, 8540504, 8540632, 8540761, 8540889, 8541018, 8541146,
                8541275, 8541403, 8541532, 8541660, 8541789, 8541917, 8542046, 8542174, 8542303,
                8411360, 8542560, 8411617, 8542817, 8411874, 8608610, 8608739, 8543331, 8543460,
                8609124, 8609253, 8609381, 8609510, 8609638, 8609767, 8609895, 8610024, 8610152,
                8610281, 8610409, 8610538, 8610666, 8610795, 8610923, 8611052, 8676716, 8676845,
                8414830, 8611566, 8415087, 8677359, 8349809, 8677616, 8350066, 8677873, 8350323,
                8678130, 8350580, 8416244, 8350837, 8678644, 8351094, 8416758, 8416887, 8417015,
                8417144, 8417272, 8417401, 8417529, 8417658, 8417786, 8417915, 8418043, 8418172,
                8418300, 8418429, 8418557, 8418686, 8353279, 8418943, 8353536, 8419200, 8353793,
                8419457, 8354050, 8419714, 7830022, 8419967, 7961347, 8747899, 7961601, 8027259,
                7896308, 7765353, 7568877, 6848092, 5865152, 5209879, 7110506, 2654167
            },
            16446, 12, 0, -32768
        );
    }

    LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
        auto inputType = cast<RankedTensorType>(op.getType(0));
        if (!inputType.getElementType().isBF16()) {
            return rewriter.notifyMatchFailure(op, "Expected bf16 output");
        }
        bool sin;
        if (isa_and_nonnull<math::SinOp>(getElementwiseUnaryOp(op)))
            sin = true;
        else if (isa_and_nonnull<math::CosOp>(getElementwiseUnaryOp(op)))
            sin = false;
        else
            return failure();

        auto maybeInvert = [&](Value invert, Value tensor) {
            auto neg =
                torq_hl::ActOp::create(
                    rewriter, op.getLoc(), inputType, createInitTensor(op, rewriter, inputType),
                    "negate", 0, 0, 0, 0, APFloat(llvm::APFloat::IEEEsingle(), "0.0"),
                    APFloat(llvm::APFloat::IEEEsingle(), "0.0"), tensor,
                    /*weights=*/mlir::Value()
                )
                    .getOutput();
            return makeSelect(op, rewriter, invert, neg, tensor);
        };

        auto [input, q34, q23] = rangeReduce(op, rewriter, op.getInputs()[0]);
        Value out;
        if (sin)
            out = maybeInvert(q34, buildSin(op, rewriter, input));
        else
            out = maybeInvert(q23, buildCos(op, rewriter, input));
        rewriter.replaceOp(op, out);
        printf("Replaced trig op\n");
        fflush(stdout);
        return success();
    }
};

} // namespace

void populateTrigPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.insert<TrigOpPattern>(context);
}

} // namespace mlir::syna::torq
